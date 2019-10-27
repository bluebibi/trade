class TrainingStopper:
    """Stops the training if the performance value doesn't improve after a given patience."""
    def __init__(self, strategy, timeframe, patience=5, delta=0, initial_performance=0.0):
        self.strategy = strategy
        self.timeframe = timeframe
        self.patience = patience
        self.counter = 0
        self.stop = False
        self.best_performance = initial_performance
        self.delta = delta

        print('INITIAL PERFORMANCE: {0:6.4f}.'.format(
            self.best_performance
        ))

    def __call__(self, current_performance, episodes_completed):
        if current_performance > self.best_performance + self.delta:
            print('[EPISODE: {0}] PERFORMANCE IMPROVED: {1:6.4f} --> {2:6.4f}. MODEL_SAVING...'.format(
                episodes_completed,
                self.best_performance,
                current_performance
            ))
            self.best_performance = current_performance
            self.counter = 0
            self.strategy.save_agent(path="./ppo_btc_{0}".format(self.timeframe))
            return False
        else:
            self.counter += 1
            print('[EPISODE: {0}] STOPPING COUNTER: {1} OUT OF {2} (BEST PERFORMANCE: {3:6.4f}, BUT NOW: {4:6.4f})'.format(
                episodes_completed,
                self.counter,
                self.patience,
                self.best_performance,
                current_performance
            ))
            if self.counter >= self.patience:
                self.stop = True
                return True
            else:
                return False
