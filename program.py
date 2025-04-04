import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import csv
from matplotlib.ticker import MultipleLocator
from scipy.optimize import curve_fit
import sys
import scipy.stats as stats


class ReadData:
    def __init__(self, file_path=None, dir_name=None):
        self.file_path = file_path
        self.dir_name = dir_name

    # ファイル名が数値のものを数値順に並べ替えて入力

    def read_data(self):
        def recursive_read(directory):
            """再帰的にディレクトリを探索してCSVファイルを読み込む"""
            data_out = []
            label_out = []
            for root, _, files in os.walk(directory):
                # CSVファイルを探して処理
                csv_files = sorted(
                    [
                        f for f in files
                        if f.endswith('.csv')
                        and f != 'table.csv'
                        and f != 'fitting_results.csv'
                        and f.rstrip('.csv').isdigit()  # ファイル名が数字か確認
                    ],
                    key=lambda x: int(x.rstrip('.csv'))  # 数値化してソート
                )
                for filename in csv_files:
                    path_file = os.path.join(root, filename)
                    # ディレクトリ名とファイル名を抽出して組み合わせる
                    relative_dir = os.path.relpath(root, directory)  # 相対パス
                    label = os.path.join(
                        relative_dir, filename) if relative_dir != '.' else filename
                    # CSVを読み込んでNaNを含む行を除去
                    df = pd.read_csv(path_file, skiprows=1).dropna()
                    label_out.append(label)
                    data_out.append(df)
            return data_out, label_out

        if self.dir_name:
            return recursive_read(self.dir_name)
        elif self.file_path:
            label_out = [os.path.basename(self.file_path)]
            data_out = [pd.read_csv(self.file_path, skiprows=1).dropna()]
            return data_out, label_out
        else:
            raise ValueError("file_pathまたはdir_nameのいずれかを指定してください。")

    def set_plot_title(self):
        pass


# データのプロットを行うクラス
class PlotData:
    def __init__(self, data, labels, save_dir):
        self.data = data
        self.labels = labels
        self.save_dir = save_dir

    # option = 1: 続けてプロットする
    # option = 0: 続けてプロットしたのちにプロットを終了する際にプロットを保存する
    # log = 0:y軸を対数にとる,1:両軸を対数にとる,2:x軸を対数スケールにとる
    # xlabel, ylabel: 軸ラベル
    # x_num,y_numはx,yで使うデータの行の数を入れる
    def plot(self, x_num, y_num, option=None, xlabel=None, ylabel=None, title=None, log=None):
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": r"\usepackage{amsmath}"
        })

        # グラフをリセットする
        if option is None:
            plt.clf()
        elif option == 0:
            plt.clf()
        df = self.data
        column = df.columns
        x = df.iloc[:, x_num]
        y = df.iloc[:, y_num]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        # 対数スケールの設定
        if log is not None:
            if len(y[y < 0]):
                print(f"Warning: {self.labels} contains negative values.")
            if log == 0:
                ax.set_yscale('log')
            elif log == 1:
                ax.set_yscale('log')
                ax.set_xscale('log')
            elif log == 2:
                ax.set_xscale('log')

        # plot
        if title is not None:
            ax.set_title(title)
        ax.plot(x, y, label=self.labels)
        if xlabel is None and ylabel is None:
            ax.set_xlabel(column[x_num])
            ax.set_ylabel(column[y_num])
        else:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        ax.grid(which='major', color='black', linestyle='-')
        ax.grid(which='minor', color='gray', linestyle='--')
        ax.legend()
        if option is None or 0:
            save_dir = os.path.join(self.save_dir, "fig")
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, f"{self.labels}_graph.pdf"))

    # ヒストグラムをプロット
    # ylabelは共通でプロットされる。

    def plot_with_hist(self, x_num, y_num, title=None,  xlabel=None, ylabel=None, log=None):
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": r"\usepackage{amsmath}"
        })

        # グラフをリセットする
        plt.clf()
        if log is not None:
            if len(y[y < 0]):
                # 対数スケールでプロットする場合に負の値があると困るのでその場合は警告を出す
                print(f"Warning: {self.labels} contains negative values.")
            if log == 0:
                axes[0].set_yscale('log')
            elif log == 1:
                axes[0].set_yscale('log')
                axes[0].set_xscale('log')
            elif log == 2:
                axes[0].set_xscale('log')
        df = self.data
        column = df.columns
        x = df.iloc[:, x_num]
        y = df.iloc[:, y_num]
        fig, axes = plt.subplots(ncols=2, sharey=True, tight_layout=True, gridspec_kw={
            'width_ratios': [3, 1]}, figsize=(12, 6))

        # 左側のグラフのプロット
        axes[0].set_xlabel(xlabel, fontsize=18)
        axes[0].plot(x, y)
        # メモリの刻み幅と最大値を定義
        yscale1 = 1
        axes[0].set_ylim([min(y)-yscale1, max(y)+yscale1])
        axes[0].yaxis.set_major_locator(MultipleLocator(yscale1))
        axes[0].grid()

        # 右側のヒストグラムのプロット
        Y, X, _ = axes[1].hist(
            y, bins=20,  orientation="horizontal")
        axes[1].set_xlabel(r"Count", fontsize=18)
        fig.supylabel(ylabel, fontsize=18)
        fig.suptitle(title, fontsize=20)
        # 分散を計算してヒストグラム上にプロット
        var = np.std(y)
        axes[1].text(
            0,  # x位置 (ヒストグラムの右側)
            max(y),  # y位置 (yの最大値に合わせて微調整)
            f"Standerd Daviation:{var:.3e}",  # 表示するテキスト
            ha='left', va='top', fontsize=13, color='blue'
        )
        fig.savefig(os.path.join(
            self.save_dir + "/hist", title + '_graph.pdf'))

    # option == None -> ディレクトリ内のデータを全てプロット
    # option == 1 -> ディレクトリ内のデータを一枚にプロット
    # option == 2 -> ディレクトリ内のヒストグラムをつけてデータを一枚にプロット
    def plot_dir(self, plot=None, x_num=None, y_num=None, xlabel=None, ylabel=None, log=None):
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": r"\usepackage{amsmath}"
        })
        if x_num is None:
            x_num = 0
        if y_num is None:
            y_num = 1
        for i in range(len(self.data)):
            self.data = self.data[i]
            self.labels = self.labels[i]
            if plot == 1:
                if i == len(self.data):
                    self.plot(x_num, y_num, option=None,
                              xlabel=xlabel, ylabel=ylabel)
                else:
                    self.plot(x_num, y_num, option=0,
                              xlabel=xlabel, ylabel=ylabel)
            elif plot == None:
                self.plot(x_num, y_num, option=None,
                          xlabel=xlabel, ylabel=ylabel)
            elif plot == 2:
                self.plot_with_hist(x_num, y_num, title=self.labels,
                                    xlabel1=xlabel, xlabel2=ylabel, ylabel=ylabel, log=log)

    # グラフにある範囲のズームを付け加える
    def plot_with_zoom(self, x_num, y_num, option=None, xlabel=None, ylabel=None, title=None, log=None):
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": r"\usepackage{amsmath}"
        })
        # ��ラフをリセットする
        if option is None:
            plt.clf()
        elif option == 0:
            plt.clf()
        df = self.data
        column = df.columns
        x = df.iloc[:, x_num]
        y = df.iloc[:, y_num]
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        # 対数スケールの設定
        if log is not None:
            if len(y[y < 0]):
                print(f"Warning: {self.labels} contains negative values.")
            if log == 0:
                ax.set_yscale('log')
            elif log == 1:
                ax.set_yscale('log')
                ax.set_xscale('log')
            elif log == 2:
                ax.set_xscale('log')

        # plot
        ax.plot(x, y, label=self.labels)
        axins = ax.inset_axes([1.10, 0.63, 0.38, 0.37])
        axins.plot(x, y, linestyle='-', label=self.labels)
        x1, x2, y1, y2 = 0, 10, 0, 0.03
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        ax.indicate_inset_zoom(axins)

        # アスペクト比の調整
        ax.set_box_aspect(0.75)
        # subplotの位置調整
        fig.subplots_adjust(left=0.11, right=0.7)

        if title is not None:
            ax.set_title(title)
        if xlabel is None and ylabel is None:
            ax.set_xlabel(column[x_num])
            ax.set_ylabel(column[y_num])
        else:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        ax.grid(which='major', color='black', linestyle='-')
        ax.grid(which='minor', color='gray', linestyle='--')
        ax.legend()

        if option is None or 0:
            save_dir = os.path.join(self.save_dir, "fig")
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(
                save_dir, f"{self.labels}_graph_with_zoom.pdf"))


# Fittignを行うクラス
class Fitting:
    def __init__(self, data, labels, save_dir):
        self.data = data
        self.labels = labels
        self.save_dir = save_dir

    # フィッティングをする関数を選択してフィッティングを行いフィッティングのパラメータを返すプログラム
    def fit_function(self, parameters, x_num, y_num, xlabel=None, ylabel=None, title=None, log=None, l=None):
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": r"\usepackage{amsmath}"
        })

        # Fitting関数の定義
        def func(x, I0, a, b, x0):
            # 特殊な場合 x = x0 で sin(a(x-x0))/(a(x-x0)) = 1 とする
            diff = x - x0
            result = np.where(
                diff == 0,  # 条件: x == x0
                I0 + b,     # 特別処理: sin(0)/0 = 1 として計算
                I0 * (np.sin(a * diff) / (a * diff))**2 + b
            )
            return result

        df = self.data
        os.makedirs(os.path.join(
            self.save_dir, "Fitting"), exist_ok=True)
        dir_path = os.path.join(self.save_dir, 'Fitting')
        with open(os.path.join(dir_path, 'fitting_results' + l + '.csv'), 'w', newline='') as log_file:
            sys.stdout = log_file
            x = df.iloc[:, x_num]
            y = df.iloc[:, y_num]
            plt.clf()

            # Fitting
            try:
                popt, pcov = curve_fit(
                    func, x, y, p0=parameters)
                x_fine = np.linspace(x.min(), x.max(), 10000)  # 1000点で補間
                fit_fine = func(x_fine, *popt)

                fig, ax = plt.subplots()
                if title is not None:
                    ax.set_title(title)
                if xlabel is None and ylabel is None:
                    column = df.columns
                    ax.set_xlabel(column[x_num])
                    ax.set_ylabel(column[y_num])
                else:
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel(ylabel)
                # ax.set_ylim(0, 1)
                ax.scatter(x, y, label='Original Data', s=15, color='black')
                # ax.plot(x, y, color='black', linestyle='-')
                ax.plot(x_fine, fit_fine, ls='-', c='blue',
                        lw=1, label="Fitted Curve")
                ax.grid(which='major', color='black', linestyle='-')
                ax.grid(which='minor', color='gray', linestyle='--')
                ax.legend()
                save_dir = os.path.join(self.save_dir, "fig")
                os.makedirs(save_dir, exist_ok=True)
                fig.savefig(os.path.join(
                    save_dir, f"{self.labels}_Fitting_graph.pdf"))
                print('label', ", ".join(map(str, parameters)))
                print(l, ", ".join(map(str, popt)))
            except RuntimeError:
                print(f"Fitting failed for {
                      self.labels}. Please adjust initial parameters or model.")
            # finally:
                # 標準出力を元に戻す
                # sys.stdout = sys.__stdout__
            return x_fine, fit_fine


# 検定を行うクラス
class Test():
    def __init__(self, filename):
        self.filename = filename

    # t検定を行う
    def t_test_confidence_interval95(self):
        df = pd.read_csv(self.filename)
        save_dir = os.path.dirname(self.filename)
        with open(os.path.join(save_dir, 'output_log.txt'), 'w') as log_file:
            try:
                data = df.iloc[:, 1]
                n = len(data)
                mean_value = np.mean(data)
                std_error = np.std(data, ddof=1)/np.sqrt(n)

                # t値の取得 (自由度 n-1, 95%信頼区間)
                t_value = stats.t.ppf(0.975, df=n-1)
                error = t_value * std_error

                lower_bound = mean_value - error
                upper_bound = mean_value + error

                sys.stdout = log_file
                print(f"サンプル数: {n}")
                print(f"平均: {mean_value}")
                print(f"標準誤差: {std_error}")
                print(f"t値 (自由度 {n-1}): {t_value}")
                print(f"95%信頼区間: ({lower_bound}, {upper_bound})")
            finally:
                # 標準出力を元に戻す
                sys.stdout = sys.__stdout__
        return lower_bound, upper_bound

# 出力データをTeX向けに変える場合に役立つプログラム


class ForTex:
    def __init__(self, dir_name):
        self.dir_name = dir_name

    # csv形式の表をTeX形式で出力する
    def table(self):
        try:
            files = [f for f in os.listdir(
                self.dir_name) if f.endswith('.csv')]
            with open(os.path.join(self.dir_name, 'table.csv'), 'w', newline='') as F:
                writer = csv.writer(F)
                for filename in files:
                    path_file = os.path.join(self.dir_name, filename)
                    writer.writerow([path_file])

                    with open(path_file) as f:
                        reader = csv.reader(f)
                        rows = list(reader)

                        for j, row in enumerate(rows):
                            string = ' & '.join(row) + ' \\\\'
                            writer.writerow([string])

                            if j == 0:
                                writer.writerow(['\\hline\\hline'])

                        writer.writerow([])
        except IOError as e:
            print(f"Error writing to CSV file: {e}")

    # ディレクトリ内のグラフをTeXで出力するコマンドを作成
    def graph(self):
        pass

    # pngファイルをpdfに変換
    def convert_png_to_pdf(self):
        try:
            files = [f for f in os.listdir(
                self.dir_name) if f.endswith('.png')]
            for filename in files:
                outname = filename.replace('.png', '.pdf')
                os.system(f'convert {filename} {outname}')
        except IOError as e:
            print(f"Error converting PNG to PDF: {e}")

# csvファイルからTeXで表を作るコードを出力する


class MakeFig:
    def __init__(self, dir_name):
        self.dir_name = dir_name

    # ディレクトリを読み込んでその中にあるcsvの表をpdfで作成する
    def make_table(self):
        files = [f for f in os.listdir(self.dir_name) if f.endswith('.csv')]
        for filename in files:
            df = pd.read_csv(filename)
            fig, ax = plt.subplots(figsize=(12, 4))

            ax.axis('off')
            ax.axis('tight')

            tb = ax.table(cellText=df.values,
                          colLabels=df.columns,
                          bbox=[0, 0, 1, 1],
                          )

            tb[0, :].set_facecolor('#363636')
            tb[0, :].set_text_props(color='w')

            plt.show()
            outname = filename.replace('.csv', '.pdf')
            plt.savefig(outname)

    # グラフを一つづつ作成する
    def make_graph(self):
        pass

    # すべてのグラフを一括して一つの画像として作成する
    def make_all_fig(self):
        pass


def main():
    pass


if __name__ == "__main__":
    main()
