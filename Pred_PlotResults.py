import warnings
warnings.filterwarnings("ignore")
from prettytable import PrettyTable
import numpy as np
import matplotlib.pyplot as plt





def stats(val):
    v = np.zeros(5)  
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    return v

def plot_results_conv():
    conv = np.load('Fitness.npy', allow_pickle=True)
    Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    Algorithm = ['TSO-ATUNet', 'BWO-ATUNet', 'CO-ATUNet', 'KOA-ATUNet', 'URP-RKOA-ATUNet']

    Value = np.zeros((conv.shape[0], 5))
    for j in range(conv.shape[0]):
        Value[j, 0] = np.min(conv[j, :])
        Value[j, 1] = np.max(conv[j, :])
        Value[j, 2] = np.mean(conv[j, :])
        Value[j, 3] = np.median(conv[j, :])
        Value[j, 4] = np.std(conv[j, :])

    Table = PrettyTable()
    Table.add_column("ALGORITHMS", Statistics)
    for j in range(len(Algorithm)):
        Table.add_column(Algorithm[j], Value[j, :])
    print('--------------------------------------------------Statistical Analysis--------------------------------------------------')
    print(Table)

    iteration = np.arange(conv.shape[1])
    plt.plot(iteration, conv[0, :], color='m', linewidth=3, marker='o', markerfacecolor='red', markersize=12,
             label='TSO-ATUNet')
    plt.plot(iteration, conv[1, :], color='c', linewidth=3, marker='p', markerfacecolor='green', markersize=12,
             label='BWO-ATUNet')
    plt.plot(iteration, conv[2, :], color='b', linewidth=3, marker='.', markerfacecolor='cyan', markersize=12,
             label='CO-ATUNet')
    plt.plot(iteration, conv[3, :], color='r', linewidth=3, marker='o', markerfacecolor='magenta', markersize=12,
             label='RKOA-ATUNet')
    plt.plot(iteration, conv[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black', markersize=12,
             label='URP-RKOA-ATUNet')
    plt.xlabel('Iteration')
    plt.ylabel('Cost Function')
    plt.legend(loc=1)
    path1 = "./Results/Conv.png"
    plt.savefig(path1)
    plt.show()



def plot_results_Pred():
    Eval_all = np.load('Eval_all.npy', allow_pickle=True)
    Terms = ['MSE','PSNR','SSIM']
    Algorithm = ['TSO-ATUNet', 'BWO-ATUNet', 'CO-ATUNet', 'RKOA-ATUNet', 'URP-RKOA-ATUNet ']
    Classifier = [ 'CNN', 'Unet ', 'Unet3+', 'ATUNet', 'URP-RKOA-ATUNet ']
    # for u in range(len(Eval_all)):
    value = Eval_all[ 4, :, :]  # Only the learning percentage 75

    Acc_Table = np.zeros((len(Algorithm)+len(Classifier), len(Terms)))
    for j in range(len(Algorithm) + len(Classifier)):
        for k in range(len(Terms)):
            Acc_Table[j, k] = Eval_all[4, j, k]
    Table = PrettyTable()
    Table.add_column('TERMS', Terms[0:])
    for k in range(len(Algorithm)):
        Table.add_column(Algorithm[k], Acc_Table[k, :])
    print('--------------------------------------------------Algorithm Comparison',
          '--------------------------------------------------')
    print(Table)
    print()

    Table = PrettyTable()
    Table.add_column('TERMS', Terms[0:])
    for k in range(len(Classifier)):
        tab = Acc_Table[k+5, :]
        Table.add_column(Classifier[k], tab)
    print('--------------------------------------------------Classifier Comparison',
          '--------------------------------------------------')
    print(Table)
    print()


    for j in range(len(Terms)):
        val = np.zeros((5, 5))
        for k in range(len(Algorithm)):
            val[k, :] = Eval_all[:, k, j]

        x = [1, 2, 3, 4, 5]

        data = val
        plt.plot(x, data[0, :], '-.',color='#65fe08', linewidth=4, marker='*', markerfacecolor='blue', markersize=13,
                 label="TSO-ATUNet")
        plt.plot(x, data[1, :],  '-.',color='#4e0550', linewidth=4, marker='*', markerfacecolor='red', markersize=13,
                 label="BWO-ATUNet")
        plt.plot(x, data[2, :],  '-.',color='#f70ffa', linewidth=4, marker='*', markerfacecolor='green', markersize=13,
                 label="CO-ATUNet")
        plt.plot(x, data[3, :], '-.',color='#a8a495', linewidth=4, marker='*', markerfacecolor='yellow', markersize=13,
                 label="RKOA-ATUNet")
        plt.plot(x, data[4, :], '-.', color='#004577', linewidth=4, marker='*', markerfacecolor='cyan', markersize=13,
                 label="URP-RKOA-ATUNet")
        plt.ylabel(Terms[j], size=16)
        plt.xticks(x, ('Linear', 'ReLU', 'Tanh', 'Softmax', 'Sigmoid'))
        plt.xlabel('Activation Function', size=16)
        # plt.legend(prop={"size": 11}, loc='best')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   ncol=3, fancybox=True, shadow=True)
        path1 = "./Results/%s_Line.png" % (Terms[j])
        plt.savefig(path1)
        plt.show()


        for j in range(len(Terms)):
            val = np.zeros((6, 5))
            for k in range(len(Classifier)):
                val[k, :] = Eval_all[ :, k+5, j]

            n_groups = 5
            data = val
            plt.subplots()
            index = np.arange(n_groups)
            bar_width = 0.10
            opacity = 1
            plt.bar(index, data[0, :], bar_width,
                    alpha=opacity,edgecolor='k', hatch='+',
                    color='#f97306',
                    label='CNN')
            plt.bar(index + bar_width, data[1, :], bar_width,
                    alpha=opacity,edgecolor='k', hatch='+',
                    color='#f10c45',
                    label='Unet ')
            plt.bar(index + bar_width + bar_width, data[2, :], bar_width,
                    alpha=opacity,
                    color='#ddd618',edgecolor='k', hatch='+',
                    label='Unet3+')
            plt.bar(index + 3 * bar_width, data[3, :], bar_width,
                    alpha=opacity,
                    color='#6ba353',edgecolor='k', hatch='+',
                    label='ATUNet')
            plt.bar(index + 4 * bar_width, data[4, :], bar_width,
                    alpha=opacity,
                    color='#13bbaf',edgecolor='w', hatch='o',
                    label='URP-RKOA-ATUNet')

            plt.xticks(index + 0.25, ('Linear', 'ReLU', 'Tanh', 'Softmax', 'Sigmoid'))
            plt.ylabel(Terms[j], size=16)
            plt.xlabel('Activation Function', size=16)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            plt.tight_layout()
            path1 = "./Results/%s_bar.png" % (Terms[j])
            plt.savefig(path1)
            plt.show()


if __name__ == '__main__':
    plot_results_Pred()
    plot_results_conv()

