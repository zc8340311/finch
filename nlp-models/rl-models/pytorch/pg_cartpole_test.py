from pg import PolicyGradient
import gym


def main():
    model = PolicyGradient(gym.make('CartPole-v1'),
                           n_in=4, n_hidden=[10], n_out=2)
    model.learn()
    model.play()

if __name__ == '__main__':
    main()
