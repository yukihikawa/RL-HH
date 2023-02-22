import numpy as np
import matplotlib.pyplot as plt

ACTOR_PATH = f"./hh_env-v0_DQN_0_MK02"
MODULE = '/recorder.npy'
recorder = np.load(ACTOR_PATH + MODULE)


def draw_learning_curve(recorder: np.ndarray = None,
                        fig_title: str = 'learning_curve1',
                        save_path: str = 'learning_curve1.jpg'):
    steps = recorder[:, 0]  # x-axis is training steps
    r_avg = recorder[:, 1]
    r_std = recorder[:, 2]
    r_exp = recorder[:, 3]
    obj_c = recorder[:, 4]
    obj_a = recorder[:, 5]

    '''plot subplots'''
    import matplotlib as mpl
    mpl.use('Agg')
    """Generating matplotlib graphs without a running X server [duplicate]
    write `mpl.use('Agg')` before `import matplotlib.pyplot as plt`
    https://stackoverflow.com/a/4935945/9293137
    """

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1)

    '''axs[0]'''
    # ax00 = axs[0]
    ax00 = axs
    ax00.cla()
    # ax01 = axs[0].twinx()
    # color01 = 'darkcyan'
    # ax01.set_ylabel('Explore AvgReward', color=color01)
    # ax01.plot(steps, r_exp, color=color01, alpha=0.5, )
    # ax01.tick_params(axis='y', labelcolor=color01)

    color0 = 'lightcoral'
    ax00.set_ylabel('Episode Return', color=color0)
    ax00.plot(steps, r_avg, label='Episode Return', color=color0)
    ax00.fill_between(steps, r_avg - r_std, r_avg + r_std, facecolor=color0, alpha=0.3)
    ax00.grid()


    '''axs[1]'''
    # ax10 = axs[1]
    # ax10.cla()

    #绘制loss
    # ax11 = axs[1].twinx()
    # color11 = 'darkcyan'
    # # ax11.set_ylabel('objC', color=color11)
    # ax11.set_ylabel('loss', color=color11)
    # ax11.fill_between(steps, obj_c, facecolor=color11, alpha=0.2, )
    # ax11.tick_params(axis='y', labelcolor=color11)

    # 绘制平均Q值
    # color10 = 'royalblue'
    # ax10.set_xlabel('Total Steps')
    # ax10.set_ylabel('objA', color=color10)
    # ax10.plot(steps, obj_a, label='objA', color=color10)
    # ax10.tick_params(axis='y', labelcolor=color10)
    # for plot_i in range(6, recorder.shape[1]):
    #     other = recorder[:, plot_i]
    #     ax10.plot(steps, other, label=f'{plot_i}', color='grey', alpha=0.5)
    # ax10.legend()
    # ax10.grid()

    '''plot save'''
    plt.title(fig_title, y=2.3)
    plt.savefig(save_path)
    plt.close('all')  # avoiding warning about too many open figures, rcParam `figure.max_open_warning`
    # plt.show()  # if use `mpl.use('Agg')` to draw figures without GUI, then plt can't plt.show()

if __name__ == '__main__':
    draw_learning_curve(recorder, fig_title='learning_curve1', save_path='learning_curve1.jpg')