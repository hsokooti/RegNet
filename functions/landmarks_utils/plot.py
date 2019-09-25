import numpy as np
import matplotlib.pyplot as plt
import functions.color_constants as color_constants


def table_tre(landmarks, exp_list, threshold=None):
    tre = dict()
    error = dict()
    fancy_name = dict()
    for exp in exp_list:
        tre[exp] = np.copy(landmarks[exp]['TRE'])
        error[exp] = np.copy(landmarks[exp]['Error'])
        fancy_name[exp] = landmarks[exp]['FancyName']
    tab = dict()
    latex_str = {}
    header_tab = ['exp', 'measure', 'measure_x', 'measure_y', 'measure_z']
    value_tab = np.empty([len(exp_list), 5], dtype=object)
    if threshold is None:
        selected_landmarks = np.arange(len(landmarks[next(iter(landmarks))]['Error']))
    else:
        affine_error = error['Affine']
        selected_landmarks = np.where(np.all(np.abs(affine_error) <= threshold, axis=1))
        selected_landmarks = selected_landmarks[0]
    selected_landmarks = selected_landmarks.astype(np.int)

    for i, exp in enumerate(exp_list):
        tab[exp] = {}
        latex_str[exp] = '&'+fancy_name[exp] + ' &test '
        tab[exp]['mean'] = np.mean(tre[exp][selected_landmarks])
        tab[exp]['std'] = np.std(tre[exp][selected_landmarks])
        value_tab[i, 0] = fancy_name[exp]
        value_tab[i, 1] = '${:.2f}\pm{:.2f}$'.format(tab[exp]['mean'], tab[exp]['std'])
        latex_str[exp] += '&\scriptsize${:.2f}\pm{:.2f}$'.format(tab[exp]['mean'], tab[exp]['std'])
        for dim in range(3):
            tab[exp]['dim' + str(dim)] = {}
            tab[exp]['dim' + str(dim)]['mean'] = np.mean(np.abs([error[exp][k][dim] for k in selected_landmarks]))
            tab[exp]['dim' + str(dim)]['std'] = np.std(np.abs([error[exp][k][dim] for k in selected_landmarks]))
            value_tab[i, dim+2] = '${:.2f}\pm{:.2f}$'.format(tab[exp]['dim' + str(dim)]['mean'], tab[exp]['dim' + str(dim)]['std'])
            latex_str[exp] += ' & ' + '\scriptsize${:.2f}\pm{:.2f}$'.format(tab[exp]['dim' + str(dim)]['mean'], tab[exp]['dim' + str(dim)]['std'])
        latex_str[exp] += r'\\'
        print(latex_str[exp])

    plt.rc('font', family='serif')
    fig, ax = plt.subplots(figsize=(15, 8))
    fig.patch.set_visible(False)
    ax.table(cellText=value_tab, colLabels=header_tab, loc='center', colLoc='left', cellLoc='left', fontsize=50)
    ax.axis('off')
    ax.axis('tight')
    fig.tight_layout()
    if threshold:
        plt.title('TRE [mm] of the capture range of {}'.format(threshold))
    plt.draw()


def boxplot_tre(landmarks, exp_list, normalize=True, min1=20, max1=90, min2=20, max2=30, dash_line_exp=None, threshold=None, ylim=None):
    tre = dict()
    error = dict()
    fancy_name = dict()
    for exp in exp_list:
        tre[exp] = np.copy(landmarks[exp]['TRE'])
        error[exp] = np.copy(landmarks[exp]['Error'])
        fancy_name[exp] = landmarks[exp]['FancyName']
    if threshold is None:
        ytick_value = [0, 5, 10, 15, min2, max2]
        ytick_label = ['0', '5', '10', '15', str(min1), str(max1)]
        title = 'TRE [mm] of all landmarks'
    else:
        affine_error = error['Affine']
        selected_landmarks = np.where(np.all(np.abs(affine_error) <= threshold, axis=1))
        for exp in exp_list:
            tre[exp] = tre[exp][selected_landmarks]
        ytick_value = [0, 5, 10, 15]
        ytick_label = [str(i) for i in ytick_value]
        while ytick_value[-1] < threshold + 5:
            ytick_value.append(ytick_value[-1] + 5)
            ytick_label.append(str(ytick_value[-2] + 5))
        title = 'TRE [mm] of the capture range of {}'.format(threshold)
    if normalize:
        for exp in exp_list:
            ix = np.where(tre[exp] > min1)
            tre[exp][ix] = (tre[exp][ix] - min1) * (max2 - min1) / (max1 - min2) + min2
    plt.rc('font', family='serif')
    fig, ax = plt.subplots(figsize=(15, 8))
    bplot1 = plt.boxplot([tre[exp] for exp in exp_list], patch_artist=True, notch=True)

    color_dict = color_constants.color_dict()
    color_keys = ['blue', 'springgreen', 'sapgreen', 'cyan2', 'peacock']
    color_list = [color_dict[color_key] for color_key in color_keys]
    for i_patch, patch in enumerate(bplot1['boxes']):
        if i_patch == 0:
            color_i = 0
        elif 0 < i_patch < 3:
            color_i = 1
        elif i_patch == 3:
            color_i = 2
        else:
            color_i = 3
        patch.set_facecolor(color_list[color_i])

    plt.xticks(np.arange(1, len(exp_list) + 1), [fancy_name[exp] for exp in exp_list], fontsize=16, rotation=90)
    plt.yticks(ytick_value, ytick_label, fontsize=24)
    if dash_line_exp is not None:
        exp_i = np.where(np.array(exp_list) == dash_line_exp)
        plt.axhline(y=bplot1['boxes'][exp_i[0][0]]._path.vertices[2,1], color='r', linestyle=':')
    if ylim is not None:
        plt.ylim([0, ylim])
    plt.subplots_adjust(hspace=0, bottom=0.5)
    plt.title(title)
    plt.draw()


def scatter_plot_dvf(landmarks, selected_exp, threshold=None, plt_limit=10):
    dvf = {selected_exp: np.copy(landmarks[selected_exp]['DVFRegNet']),
           'GroundTruth': np.copy(landmarks['Affine']['GroundTruth'])}
    if threshold is None:
        selected_landmarks = np.arange(len(landmarks['Affine']['Error']))
    else:
        affine_error = landmarks['Affine']['Error']
        selected_landmarks = np.where(np.all(np.abs(affine_error) <= threshold, axis=1))
        plt_limit = threshold
    for exp in dvf.keys():
        dvf[exp] = dvf[exp][selected_landmarks]
    plt.rc('font', family='serif')
    for dim in range(3):
        plt.figure(figsize=(6, 6))
        plt.scatter(dvf['GroundTruth'][:, dim], dvf[selected_exp][:, dim], color='green')
        plt.ylim((-plt_limit-5, plt_limit+5))
        plt.xlim((-plt_limit-5, plt_limit+5))
        plt.title('Dimension'+str(dim))
        plt.xlabel('Ground truth [mm]', fontsize=20)
        plt.ylabel(landmarks[selected_exp]['FancyName'] + ' [mm]', fontsize=20)
        plt.plot([-plt_limit-5, plt_limit+5], [-plt_limit-5, plt_limit+5], color='blue', linewidth=2)
        plt.draw()


def scatter_plot_tre(landmarks, selected_exp, normalize=True, min1=25, max1=90, min2=25, max2=40):
    dvf_selected = np.copy(landmarks[selected_exp]['DVFRegNet'])
    dvf_magnitude = np.array([np.linalg.norm(dvf_selected[i, :]) for i in range(np.shape(landmarks[selected_exp]['DVFRegNet'])[0])])
    tre = {selected_exp: dvf_magnitude,
           'GroundTruth': np.copy(landmarks['Affine']['TRE'])}
    if normalize:
        for exp in tre.keys():
            ix = np.where(tre[exp] > min1)
            tre[exp][ix] = (tre[exp][ix] - min1) * (max2 - min1) / (max1 - min2) + min2
    plt.rc('font', family='serif')
    plt.figure(figsize=(12, 6))
    plt.plot(tre['GroundTruth'], tre[selected_exp], 'o', color='cyan')
    xTickPoints = np.arange(0, min1+5, 5)
    xTickLabels = [str(myNum) for myNum in xTickPoints]
    xTickPoints = np.append(xTickPoints, max2)
    xTickLabels.append(str(max1))
    yTickPoints = np.arange(0, min1+5, 5)
    plt.xticks(xTickPoints, xTickLabels , fontsize=20, rotation=30)
    plt.yticks(yTickPoints, [str(myNum) for myNum in yTickPoints], fontsize=20, rotation=90)
    plt.plot([0, min1],[0, min2], color='blue')
    plt.ylim([0, max(tre['GroundTruth'])])
    plt.xlim([0, xTickPoints[-1]])
    plt.xlabel('Ground truth [mm]', fontsize=20)
    plt.ylabel(landmarks[selected_exp]['FancyName'] + ' [mm]', fontsize=20)
    plt.subplots_adjust(hspace=0, bottom=0.2)
    plt.draw()
