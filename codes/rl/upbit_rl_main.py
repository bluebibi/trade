import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=ConvergenceWarning)

import sys,os

idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade"
sys.path.append(PROJECT_HOME)

from common.global_variables import SLACK_WEBHOOK_URL_1, SLACK_WEBHOOK_URL_2, SOURCE
from codes.rl.upbit_rl_constants import MAX_EPISODES, REPLAY_MEMORY_THRESHOLD_FOR_TRAIN, TRAIN_INTERVAL_STEPS, \
    QNET_COPY_TO_TARGET_QNET_INTERVAL_EPISODES, QNET_COPY_TO_TARGET_QNET_INTERVAL_STEPS, EPSILON_START, \
    EPSILON_FINAL, MODEL_SAVE_PATH, PERFORMANCE_SAVE_PATH
from codes.rl.upbit_rl_env import UpbitEnvironment
from codes.rl.upbit_rl_policy import DeepBuyerPolicy, DeepSellerPolicy
from codes.rl.upbit_rl_utils import print_before_step, print_after_step, EnvironmentType, EnvironmentStatus, \
    BuyerAction, SellerAction, draw_performance, save_performance
from common.slack import PushSlack
import argparse
import numpy as np

pusher = PushSlack(SLACK_WEBHOOK_URL_1, SLACK_WEBHOOK_URL_2)


def linearly_decaying_epsilon(max_episodes, episode, warmup_episodes, max_epsilon, min_epsilon):
    steps_left = max_episodes + warmup_episodes - episode
    bonus = (max_epsilon - min_epsilon) * steps_left / max_episodes
    bonus = np.clip(bonus, 0., max_epsilon - min_epsilon)
    return min_epsilon + bonus


def main(args):
    coin_name = args.coin
    env = UpbitEnvironment(coin_name=coin_name, args=args, env_type=EnvironmentType.TRAIN)
    buyer_policy = DeepBuyerPolicy(args)
    seller_policy = DeepSellerPolicy(args)

    beta_start = 0.4
    beta_frames = 1000
    beta_by_episode = lambda episode: min(1.0, beta_start + episode * (1.0 - beta_start) / beta_frames)

    START_EPISODE = int(env.last_episode) + 1

    total_balance_per_episode = 0.0

    for episode in range(START_EPISODE, MAX_EPISODES):
        done = False
        num_steps = 0
        epsilon = linearly_decaying_epsilon(
            max_episodes=MAX_EPISODES,
            episode=episode,
            warmup_episodes=5,
            max_epsilon=EPSILON_START,
            min_epsilon=EPSILON_FINAL
        )
        observation, info_dic = env.reset(
            episode, epsilon, buyer_policy, seller_policy
        )

        from_buy_model = 0
        score = 0.0
        beta = beta_by_episode(episode)

        while not done:
            print_before_step(env, coin_name, episode, MAX_EPISODES, num_steps, env.total_steps, info_dic)

            if env.status is EnvironmentStatus.TRYING_BUY:
                action, from_buy_model = buyer_policy.sample_action(observation, info_dic, epsilon)

                next_observation, reward, done, next_info_dic = env.step_with_info_dic(action, info_dic)

                if action is BuyerAction.MARKET_BUY:
                    done_mask = 0.0
                    action = 1
                    buyer_policy.pending_buyer_transition = [episode, observation, action, None, None, done_mask]
                    next_env_state = EnvironmentStatus.TRYING_SELL

                    env.market_buys += 1

                    if from_buy_model:
                        env.market_buys_from_model += 1
                else:
                    done_mask = 0.0 if done else 1.0
                    action = 0
                    buyer_policy.buyer_memory.put([episode, observation, action, reward, next_observation, done_mask])

                    score += reward

                    next_env_state = EnvironmentStatus.TRYING_BUY

            elif env.status is EnvironmentStatus.TRYING_SELL:
                action, from_sell_model = seller_policy.sample_action(observation, info_dic, epsilon)

                next_observation, reward, done, next_info_dic = env.step_with_info_dic(action, info_dic)

                if action is SellerAction.MARKET_SELL:
                    buyer_policy.pending_buyer_transition[3] = reward * 10 if reward > 0.0 else reward
                    buyer_policy.pending_buyer_transition[4] = next_observation
                    buyer_policy.buyer_memory.put(buyer_policy.pending_buyer_transition)
                    buyer_policy.pending_buyer_transition = None
                    score += reward

                    done_mask = 0.0
                    action = 1
                    seller_policy.seller_memory.put([
                        episode, observation, action,
                        reward * 10 if reward > 0.0 else reward,
                        next_observation, done_mask
                    ])
                    next_env_state = EnvironmentStatus.TRYING_BUY
                    score += reward

                    env.market_sells += 1

                    if from_sell_model:
                        env.market_sells_from_model += 1

                    if reward > 0.0:
                        env.market_profitable_sells += 1
                        env.market_profitable_buys += 1
                        if from_sell_model:
                            env.market_profitable_sells_from_model += 1
                            if from_buy_model:
                                env.market_profitable_buys_from_model += 1
                        else:
                            if from_buy_model:
                                env.market_profitable_buys_from_model += 1
                else:
                    done_mask = 0.0 if done else 1.0
                    action = 0
                    seller_policy.seller_memory.put([episode, observation, action, reward, next_observation, done_mask])
                    score += reward
                    next_env_state = EnvironmentStatus.TRYING_SELL
            else:
                raise ValueError("Environment Status Error: {0}".format(env.status))

            num_steps += 1

            print_after_step(env, action, next_observation, reward, buyer_policy, seller_policy, epsilon, num_steps,
                             episode, done)

            if not args.train_episode_ends:
                # Replay Memory 저장 샘플이 충분하다면 buyer_policy 또는 seller_policy 강화학습 훈련 (딥러닝 모델 최적화)
                if num_steps % TRAIN_INTERVAL_STEPS == 0 or done:
                    if buyer_policy.buyer_memory.size() >= REPLAY_MEMORY_THRESHOLD_FOR_TRAIN:
                        _ = buyer_policy.train(beta)
                    if seller_policy.seller_memory.size() >= REPLAY_MEMORY_THRESHOLD_FOR_TRAIN:
                        _ = seller_policy.train(beta)

                #TARGET Q Network 으로 Q Network 파라미터 Copy
                if num_steps % QNET_COPY_TO_TARGET_QNET_INTERVAL_STEPS == 0:
                    buyer_policy.qnet_copy_to_target_qnet()
                    seller_policy.qnet_copy_to_target_qnet()

            if done:
                buyer_loss = buyer_policy.train(beta)
                seller_loss = seller_policy.train(beta)

                if episode != 0 and episode % QNET_COPY_TO_TARGET_QNET_INTERVAL_EPISODES == 0:
                    buyer_policy.qnet_copy_to_target_qnet()
                    seller_policy.qnet_copy_to_target_qnet()

                total_balance_per_episode = env.balance + env.hold_coin_krw

                # 성능 그래프 그리기
                env.buyer_loss_list.append(buyer_loss)
                env.seller_loss_list.append(seller_loss)
                env.total_balance_per_episode_list.append(total_balance_per_episode)

                env.market_buy_list.append(env.market_buys)
                env.market_sell_list.append(env.market_sells)
                env.market_buy_by_model_list.append(env.market_buys_from_model)
                env.market_sell_by_model_list.append(env.market_sells_from_model)
                env.market_profitable_buy_list.append(env.market_profitable_buys)
                env.market_profitable_sell_list.append(env.market_profitable_sells)
                env.market_profitable_buy_by_model_list.append(env.market_profitable_buys_from_model)
                env.market_profitable_sell_by_model_list.append(env.market_profitable_sells_from_model)

                env.total_profit_list.append(env.total_profit)
                env.score_list.append(score)

                save_performance(env)
                draw_performance(env, args)

            # 다음 스텝 수행을 위한 사전 준비
            observation = next_observation
            info_dic = next_info_dic
            env.status = next_env_state

        market_profitable_buys_from_model_rate = env.market_profitable_buys_from_model / env.market_buys_from_model if env.market_buys_from_model != 0 else 0.0
        market_profitable_sells_from_model_rate = env.market_profitable_sells_from_model / env.market_sells_from_model if env.market_sells_from_model != 0 else 0.0

        model_save_condition_list = [
            total_balance_per_episode >= env.max_total_balance_per_episode,
            env.market_profitable_buys_from_model != 0,
            env.market_profitable_sells_from_model != 0
        ]

        if all(model_save_condition_list):
            env.max_total_balance_per_episode = total_balance_per_episode

            buyer_policy.save_model(
                episode=episode,
                max_total_balance_per_episode=env.max_total_balance_per_episode,
                market_profitable_buys_from_model_rate=market_profitable_buys_from_model_rate,
                market_profitable_sells_from_model_rate=market_profitable_sells_from_model_rate
            )

            seller_policy.save_model(
                episode=episode,
                max_total_balance_per_episode=env.max_total_balance_per_episode,
                market_profitable_buys_from_model_rate=market_profitable_buys_from_model_rate,
                market_profitable_sells_from_model_rate=market_profitable_sells_from_model_rate
            )

            print("NEW MODEL SAVED AT {0} EPISODE".format(episode))

            if args.slack:
                pusher.send_message("me", "[{0}] {1}, {2}/{3}, {4}/{5}, {6:8}, {7:8}, {8}/{9}={10:5.3f}, {11}/{12}={13:5.3f}".format(
                    SOURCE,
                    coin_name,
                    episode, MAX_EPISODES,
                    num_steps, env.total_steps,
                    env.max_total_balance_per_episode,
                    env.score_list[-1],
                    env.market_profitable_buys_from_model, env.market_buys_from_model,
                    market_profitable_buys_from_model_rate,
                    env.market_profitable_sells_from_model, env.market_sells_from_model,
                    market_profitable_sells_from_model_rate
                ))
        env.last_episode = episode


if __name__ == "__main__":
    ##
    ## python upbit_rl_main.py -p -v -e -u -data_limit=3000 -hold_reward=-0.000 -window_size=36 -coin=BTC
    ##
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--per', action='store_true', help="use prioritized experience memory")
    parser.add_argument('-f', '--federated', action='store_true', help="use federated learning")
    parser.add_argument('-l', '--lstm', action='store_true', help="use LSTM (default CNN)")
    parser.add_argument('-e', '--train_episode_ends', action='store_true', help="train only when an episode ends")
    parser.add_argument('-s', '--slack', action='store_true', help="slack message when an episode ends")
    parser.add_argument('-u', '--pseudo', action='store_true', help="pseudo rl data")
    parser.add_argument('-o', '--ohlc', action='store_true', help="Open-high-low-close")
    parser.add_argument('-v', '--volume', action='store_true', help="use volume information in order book")
    parser.add_argument('-hold_reward', required=True, help="hold reward")
    parser.add_argument('-window_size', required=True, help="window size")
    parser.add_argument('-data_limit', required=True, help="data_limit")
    parser.add_argument('-coin', required=True, help="coin name")
    args = parser.parse_args()

    import os

    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)

    if not os.path.exists(PERFORMANCE_SAVE_PATH):
        os.makedirs(PERFORMANCE_SAVE_PATH)

    main(args)

    #upbit = Upbit(CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt)
    #coin_names = upbit.get_all_coin_names()
    # for coin_name in coin_names:
    #     main(coin_name, args)