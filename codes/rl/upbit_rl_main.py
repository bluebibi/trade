import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=ConvergenceWarning)

import sys,os

idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade"
sys.path.append(PROJECT_HOME)

from common.global_variables import CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt, SLACK_WEBHOOK_URL_1, SLACK_WEBHOOK_URL_2, \
    SOURCE
from codes.rl.upbit_rl_constants import MAX_EPISODES, \
    REPLAY_MEMORY_THRESHOLD_FOR_TRAIN, TRAIN_INTERVAL, QNET_COPY_TO_TARGET_QNET_INTERVAL, EPSILON_START, \
    PERFORMANCE_GRAPH_DRAW_INTERVAL, SAVE_MODEL_INTERVAL, EPSILON_FINAL
from codes.rl.upbit_rl_env import UpbitEnvironment
from codes.upbit.upbit_api import Upbit
from codes.rl.upbit_rl_policy import DeepBuyerPolicy, DeepSellerPolicy
from codes.rl.upbit_rl_utils import print_before_step, print_after_step, EnvironmentType, \
    EnvironmentStatus, BuyerAction, SellerAction, draw_performance
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
    env = UpbitEnvironment(coin_name=coin_name, args=args, env_type=EnvironmentType.TRAIN_VALID)
    buyer_policy = DeepBuyerPolicy(args)
    seller_policy = DeepSellerPolicy(args)

    total_profit_list = []
    buyer_loss_list = []
    seller_loss_list = []
    market_buy_list = []
    market_sell_list = []
    market_profitable_buy_list = []
    market_profitable_sell_list = []

    market_buy_from_model_list = []
    market_sell_from_model_list = []
    market_profitable_buy_from_model_list = []
    market_profitable_sell_from_model_list = []

    beta_start = 0.4
    beta_frames = 1000
    beta_by_episode = lambda episode: min(1.0, beta_start + episode * (1.0 - beta_start) / beta_frames)

    for episode in range(MAX_EPISODES):
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
        buyer_loss = 0.0
        seller_loss = 0.0
        market_buys = 0
        market_sells = 0
        market_profitable_buys = 0
        market_profitable_sells = 0

        market_buys_from_model = 0
        market_sells_from_model = 0
        market_profitable_buys_from_model = 0
        market_profitable_sells_from_model = 0

        from_buy_model = 0

        while not done:
            print_before_step(env, coin_name, episode, MAX_EPISODES, num_steps, env.total_steps, info_dic)

            if env.status is EnvironmentStatus.TRYING_BUY:
                action, from_buy_model = buyer_policy.sample_action(observation, info_dic, epsilon)

                next_observation, reward, done, next_info_dic = env.step_with_info_dic(action, info_dic)

                if action is BuyerAction.MARKET_BUY:
                    action = 1
                    buyer_policy.pending_buyer_transition = [observation, action, None, None, 0.0]
                    next_env_state = EnvironmentStatus.TRYING_SELL

                    market_buys += 1

                    if from_buy_model:
                        market_buys_from_model += 1
                else:
                    done_mask = 0.0 if done else 1.0
                    action = 0
                    buyer_policy.buyer_memory.put((observation, action, reward, next_observation, done_mask))
                    next_env_state = EnvironmentStatus.TRYING_BUY

            elif env.status is EnvironmentStatus.TRYING_SELL:
                action, from_sell_model = seller_policy.sample_action(observation, info_dic, epsilon)

                next_observation, reward, done, next_info_dic = env.step_with_info_dic(action, info_dic)

                if action is SellerAction.MARKET_SELL:
                    done_mask = 0.0 if done else 1.0

                    buyer_policy.pending_buyer_transition[2] = reward
                    buyer_policy.pending_buyer_transition[3] = next_observation
                    # buyer_policy.pending_buyer_transition[4] = done_mask
                    buyer_policy.buyer_memory.put(tuple(buyer_policy.pending_buyer_transition))
                    buyer_policy.pending_buyer_transition = None

                    # done_mask = 0.0 if done else 1.0
                    action = 1
                    seller_policy.seller_memory.put((observation, action, reward, next_observation, 0.0))
                    next_env_state = EnvironmentStatus.TRYING_BUY

                    market_sells += 1

                    if from_sell_model:
                        market_sells_from_model += 1

                    if reward > 0.0:
                        market_profitable_sells += 1
                        market_profitable_buys += 1
                        if from_sell_model:
                            market_profitable_sells_from_model += 1
                            if from_buy_model:
                                market_profitable_buys_from_model += 1
                else:
                    done_mask = 0.0 if done else 1.0
                    action = 0
                    seller_policy.seller_memory.put((observation, action, reward, next_observation, done_mask))
                    next_env_state = EnvironmentStatus.TRYING_SELL

            else:
                raise ValueError("Environment Status Error: {0}".format(env.status))

            num_steps += 1
            if not done:
                print_after_step(env, action, next_observation, reward, done, buyer_policy, seller_policy, epsilon)

            # Replay Memory 저장 샘플이 충분하다면 buyer_policy 또는 seller_policy 강화학습 훈련 (딥러닝 모델 최적화)
            if num_steps % TRAIN_INTERVAL == 0 or done:
                beta = beta_by_episode(episode)
                if buyer_policy.buyer_memory.size() >= REPLAY_MEMORY_THRESHOLD_FOR_TRAIN:
                    #buyer_policy.load_model()
                    buyer_loss = buyer_policy.train(beta)
                    #buyer_policy.save_model()
                if seller_policy.seller_memory.size() >= REPLAY_MEMORY_THRESHOLD_FOR_TRAIN:
                    #seller_policy.load_model()
                    seller_loss = seller_policy.train(beta)
                    #seller_policy.save_model()

                # AWS S3로 모델 저장
                if num_steps % SAVE_MODEL_INTERVAL == 0 or done:
                    buyer_policy.save_model()
                    seller_policy.save_model()

            # TARGET Q Network 으로 Q Network 파라미터 Copy
            if num_steps % QNET_COPY_TO_TARGET_QNET_INTERVAL == 0:
                buyer_policy.qnet_copy_to_target_qnet()
                seller_policy.qnet_copy_to_target_qnet()


            # 성능 그래프 그리기
            total_profit_list.append(env.total_profit)
            buyer_loss_list.append(buyer_loss)
            seller_loss_list.append(seller_loss)
            market_buy_list.append(market_buys)
            market_sell_list.append(market_sells)
            market_buy_from_model_list.append(market_buys_from_model)
            market_sell_from_model_list.append(market_sells_from_model)
            market_profitable_buy_list.append(market_profitable_buys)
            market_profitable_sell_list.append(market_profitable_sells)
            market_profitable_buy_from_model_list.append(market_profitable_buys_from_model)
            market_profitable_sell_from_model_list.append(market_profitable_sells_from_model)

            if num_steps % PERFORMANCE_GRAPH_DRAW_INTERVAL == 0 or done:
                draw_performance(
                    total_profit_list, buyer_loss_list, seller_loss_list, market_buy_list, market_sell_list,
                    market_buy_from_model_list, market_sell_from_model_list,
                    market_profitable_buy_list, market_profitable_sell_list,
                    market_profitable_buy_from_model_list, market_profitable_sell_from_model_list,
                    args
                )

            # 다음 스텝 수행을 위한 사전 준비
            observation = next_observation
            info_dic = next_info_dic
            env.status = next_env_state

            if env.balance <= 0.0:
                done = True

        pusher.send_message("me", "[{0}] {1}, {2}/{3}, {4}/{5}, {6}".format(
            SOURCE,
            coin_name,
            episode + 1, MAX_EPISODES,
            num_steps, env.total_steps,
            0.0 if env.balance <= 0.0 else env.balance
        ))


if __name__ == "__main__":
    ##
    ## python upbit_rl_main.py -p -coin=BTC
    ##
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--per', action='store_true', help="use prioritized experience memory")
    parser.add_argument('-f', '--federated', action='store_true', help="use federated learning")
    parser.add_argument('-l', '--lstm', action='store_true', help="use LSTM (default CNN)")
    parser.add_argument('-v', '--volume', action='store_true', help="use volume information in order book")
    parser.add_argument('-coin', required=True, help="coin name")
    args = parser.parse_args()

    main(args)

    #upbit = Upbit(CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt)
    #coin_names = upbit.get_all_coin_names()
    # for coin_name in coin_names:
    #     main(coin_name, args)