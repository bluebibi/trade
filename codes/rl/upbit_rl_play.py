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
from codes.rl.upbit_rl_constants import MAX_EPISODES, MODEL_SAVE_PATH, PLAY_SAVE_PATH
from codes.rl.upbit_rl_env import UpbitEnvironment
from codes.rl.upbit_rl_policy import DeepBuyerPolicy, DeepSellerPolicy
from codes.rl.upbit_rl_utils import print_before_step, print_after_step, EnvironmentType, EnvironmentStatus, \
    BuyerAction, SellerAction, draw_performance, draw_play_record
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
    env = UpbitEnvironment(coin_name=coin_name, args=args, env_type=EnvironmentType.VALID)
    buyer_policy = DeepBuyerPolicy(args, play=True)
    seller_policy = DeepSellerPolicy(args, play=True)

    env.total_balance_per_episode_list.clear()

    env.total_profit_list.clear()
    env.buyer_loss_list.clear()
    env.seller_loss_list.clear()
    env.market_buy_list.clear()
    env.market_sell_list.clear()
    env.market_profitable_buy_list.clear()
    env.market_profitable_sell_list.clear()

    env.market_buy_by_model_list.clear()
    env.market_sell_by_model_list.clear()
    env.market_profitable_buy_by_model_list.clear()
    env.market_profitable_sell_by_model_list.clear()

    total_balance_per_episode = 0.0

    episode = 1
    epsilon = 0.0

    observation, info_dic = env.reset(episode, epsilon, buyer_policy, seller_policy)

    from_buy_model = 0
    score = 0.0

    num_steps = 0
    done = False

    coin_unit_price_list = [info_dic["coin_unit_price"]]
    market_buy_step_list = []
    market_sell_step_list = []

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

                market_buy_step_list.append((num_steps, info_dic["coin_unit_price"]))
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

                market_sell_step_list.append((num_steps, info_dic["coin_unit_price"]))
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

        # 다음 스텝 수행을 위한 사전 준비
        observation = next_observation
        info_dic = next_info_dic
        env.status = next_env_state

        total_balance_per_episode = env.balance + env.hold_coin_krw
        coin_unit_price_list.append(info_dic["coin_unit_price"])

    draw_play_record(coin_unit_price_list, market_buy_step_list, market_sell_step_list)

    market_profitable_buys_from_model_rate = env.market_profitable_buys_from_model / env.market_buys_from_model if env.market_buys_from_model != 0 else 0.0
    market_profitable_sells_from_model_rate = env.market_profitable_sells_from_model / env.market_sells_from_model if env.market_sells_from_model != 0 else 0.0

    if args.slack:
        pusher.send_message("me", "[{0}] {1}, {2}/{3}, {4}/{5}, {6}, {7}/{8}={9:5.3f}, {10}/{11}={12:5.3f}".format(
            SOURCE, coin_name,
            episode, MAX_EPISODES,
            num_steps, env.total_steps,
            total_balance_per_episode,
            env.market_profitable_buys_from_model, env.market_buys_from_model,
            market_profitable_buys_from_model_rate,
            env.market_profitable_sells_from_model, env.market_sells_from_model,
            market_profitable_sells_from_model_rate
        ))


if __name__ == "__main__":
    ##
    ## python upbit_rl_play.py -v -s -u -window_size=36 -data_limit=10000 -hold_reward=-0.002 -coin=BTC
    ##
    parser = argparse.ArgumentParser()

    parser.add_argument('-l', '--lstm', action='store_true', help="use LSTM (default CNN)")
    parser.add_argument('-v', '--volume', action='store_true', help="use volume information in order book")
    parser.add_argument('-s', '--slack', action='store_true', help="slack message when an episode ends")
    parser.add_argument('-u', '--pseudo', action='store_true', help="pseudo rl data")
    parser.add_argument('-o', '--ohlc', action='store_true', help="Open-high-low-close")
    parser.add_argument('-window_size', required=True, help="window size")
    parser.add_argument('-hold_reward', required=True, help="hold reward")
    parser.add_argument('-data_limit', required=True, help="data_limit")
    parser.add_argument('-coin', required=True, help="coin name")
    args = parser.parse_args()

    import os

    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)

    if not os.path.exists(PLAY_SAVE_PATH):
        os.makedirs(PLAY_SAVE_PATH)

    main(args)