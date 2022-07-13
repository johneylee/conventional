class Cls:
    def __init__(self):
        self.alpha = {'0': 0.92, 's': 0.9, 'd': 0.85}
        self.s = {'hat': 1, 'f': 1}

    def eq0(self):
        s_hat = self.alpha['s'] * self.s['hat'] + (1 - self.alpha['s'] * self.s['f'])

        self.s['hat'] = s_hat

    def eq1(self):
        pass

    def eq2(self):
        pass

    def noise_estimation(self):
        self.eq0()
        for i in range(100):
            self.eq1()
        self.eq2()

        return noise_PSD


if __name__ == "__main__":
    a = Cls()
    noise_PSD = a.noise_estimation()