import csv
import math
from datetime import datetime
from collections import OrderedDict
from decimal import Decimal
# import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import timeit
import os
import logging
from tqdm import trange

def timestamp2datetime(timeStamp):
    try:
        d = datetime.fromtimestamp(int(timeStamp))
        str1 = d.strftime("%Y-%m-%d %H:%M:%S.%f")
        # 2015-08-28 16:43:37.283000'
        return str1
    except Exception as e:
        print(e)
        return ''

# 计时装饰器
def clock(func):
    def clocked(*args):
        t0 = timeit.default_timer()
        result = func(*args)
        elapsed = timeit.default_timer() - t0
        logging.debug('Time_cost:[%0.8fs]' % elapsed)
        return result

    return clocked

def n2s(num):
    if num <= 9:
        return '0' + str(num)
    else:
        return str(num)

def rss_crd(tra_filename):
    # get raw data from .csv file

    # get rss

    fp_coor = {}

    with open(tra_filename) as f:
        reader = list(csv.reader(f))
        fp_len = len(reader[0])
        for i in range(len(reader)):
            if i % 6 == 0:
                continue
            if i % 6 == 1:
                fp = fp_len * [0]
            for j in range(fp_len):
                if reader[i][j] == '100':
                    fp[j] = fp[j] - 105
                else:
                    fp[j] = fp[j] + int(reader[i][j])
            if i % 6 == 5:
                for j in range(fp_len):
                    fp[j] = fp[j] // 5
                    # if fp[j] == -100:
                    #    fp[j] = 100
                fp_coor['rp' + str(i // 6)] = fp

    # get crd match fp

    crd_filename = tra_filename.split('.')[0][:-3] + 'crd.csv'
    with open(crd_filename) as crd:
        coor = list(csv.reader(crd))
        crd_len = len(coor)
        for i in range(0, crd_len, 6):
            fp_coor['rp' + str(i // 6)] = fp_coor['rp' + str(i // 6)] + coor[i]

    return fp_coor

# 引入RMSE 剔除粗大误差
# def rss_crd(tra_filename):
#     """
#     :param tra_filename: 训练集文件名
#     :return: 指纹库
#     """
#     # get raw data from .csv file
#
#     # get rss
#
#     fp_coor = {}
#
#     with open(tra_filename) as f:
#         reader = list(csv.reader(f))
#         r_len = len(reader)
#         fp_len = len(reader[0])
#         rss = np.array(list(map(int, reader[0])))
#         for i in range(1, r_len):
#             rss = np.vstack((rss, list(map(int, reader[i]))))
#         rss[rss == 100] = -105
#
#     mean_rss = [0] * r_len
#     for i in range(0, r_len, 6):
#         _mean_rss = np.mean(rss[i:i + 6, :], 0)
#         if i == 0:
#             mean_rss = _mean_rss
#         else:
#             mean_rss = np.vstack((mean_rss, _mean_rss))
#
#     # RMSE 均方根误差
#     # v 残差矩阵
#     v = [0] * fp_len
#     for i in range(r_len):
#         _v = (rss[i] - mean_rss[i // 6]) ** 2
#         if i == 0:
#             v = _v
#         else:
#             v = np.vstack((v, _v))
#     # sigma 均方根误差矩阵
#     sigma = [0] * r_len
#     for i in range(0, r_len, 6):
#         _sigma = np.sqrt(0.2 * np.sum(v[i:i + 6, :], 0))
#         if i == 0:
#             sigma = _sigma
#         else:
#             sigma = np.vstack((sigma, _sigma))
#     three_sigma = 3 * sigma
#
#     v = np.sqrt(v)
#
#     fv = [0] * r_len
#     for i in range(r_len):
#         _v = v[i] - three_sigma[i // 6]
#         if i == 0:
#             fv = _v
#         else:
#             fv = np.vstack((fv, _v))
#     # v < 3 * simga 保留 存为1
#     fv[fv >= 0] = 0
#     fv[fv < 0] = 1
#     # 对RSS进行过滤
#     f_rss = fv * rss
#
#     # 求均值 形成最终的RSS指纹
#     final_rss = [0] * r_len
#     count = [0] * r_len
#     for i in range(0, r_len, 6):
#         _f_rss = np.sum(f_rss[i:i + 6, :], 0)
#         _count = np.sum(fv[i:i + 6, :], 0)
#         if i == 0:
#             final_rss = _f_rss
#             count = _count
#         else:
#             final_rss = np.vstack((final_rss, _f_rss))
#             count = np.vstack((count, _count))
#     final_rss = final_rss / count
#
#     where_are_nan = np.isnan(final_rss)
#     final_rss[where_are_nan] = -105
#
#     for i in range(0, r_len, 6):
#         fp_coor['rp' + str(i // 6)] = list(map(str, final_rss[i // 6, :]))
#
#     crd_filename = tra_filename.split('.')[0][:-3] + 'crd.csv'
#     with open(crd_filename) as crd:
#         coor = list(csv.reader(crd))
#         crd_len = len(coor)
#         for i in range(0, crd_len, 6):
#             fp_coor['rp' + str(i // 6)] = fp_coor['rp' + str(i // 6)] + coor[i]
#
#     return fp_coor

# def rss_crd(tra_filename):
#     # get raw data from .csv file
#
#     # get rss
#
#     fp_coor = {}
#
#     with open(tra_filename) as f:
#         reader = list(csv.reader(f))
#         fp_len = len(reader[0])
#         for i in range(len(reader)):
#             if i % 6 == 0:
#                 fp = fp_len * [0]
#             for j in range(fp_len):
#                 if reader[i][j] == '100':
#                     fp[j] = fp[j] - 105
#                 else:
#                     fp[j] = fp[j] + int(reader[i][j])
#             if i % 6 == 5:
#                 for j in range(fp_len):
#                     fp[j] = fp[j] // 6
#                     # if fp[j] == -100:
#                     #    fp[j] = 100
#                 fp_coor['rp' + str(i // 6)] = fp
#
#     # get crd match fp
#
#     crd_filename = tra_filename.split('.')[0][:-3] + 'crd.csv'
#     with open(crd_filename) as crd:
#         coor = list(csv.reader(crd))
#         crd_len = len(coor)
#         for i in range(0, crd_len, 6):
#             fp_coor['rp' + str(i // 6)] = fp_coor['rp' + str(i // 6)] + coor[i]
#
#     return fp_coor

def rss_crd_filter(tra_filename):
    """
    :param tra_filename: 训练集文件名
    :return: 指纹库
    """
    # get raw data from .csv file

    # get rss

    fp_coor = {}

    with open(tra_filename) as f:
        reader = list(csv.reader(f))
        r_len = len(reader)
        fp_len = len(reader[0])
        rss = np.array(list(map(int, reader[0])))
        for i in range(1, r_len):
            rss = np.vstack((rss, list(map(int, reader[i]))))
        rss[rss == 100] = -105

    up_rss = [0] * r_len
    down_rss = [0] * r_len
    for i in range(0, r_len, 6):
        mean_rss = np.mean(rss[i:i + 6, :], 0)
        std_rss = np.std(rss[i:i + 6, :], 0)
        if i == 0:
            up_rss = mean_rss + std_rss
            down_rss = mean_rss - std_rss
        else:
            _up_rss = mean_rss + std_rss
            _down_rss = mean_rss - std_rss
            up_rss = np.vstack((up_rss, _up_rss))
            down_rss = np.vstack((down_rss, _down_rss))

    for i in range(r_len):
        index = i // 6
        if i % 6 == 0:
            fp = [0] * fp_len
            count = [0] * fp_len
        for j in range(fp_len):
            if down_rss[index][j] <= rss[i][j] <= up_rss[index][j]:
                count[j] = count[j] + 1
                fp[j] = fp[j] + rss[i][j]
        if i % 6 == 5:
            for j in range(fp_len):
                fp[j] = fp[j] / count[j]
            fp_coor['rp' + str(i // 6)] = fp

    crd_filename = tra_filename.split('.')[0][:-3] + 'crd.csv'
    with open(crd_filename) as crd:
        coor = list(csv.reader(crd))
        crd_len = len(coor)
        for i in range(0, crd_len, 6):
            fp_coor['rp' + str(i // 6)] = fp_coor['rp' + str(i // 6)] + coor[i]

    return fp_coor

def floor_filter(tra, floor):
    f_c_tra = {}
    for rp, fp in tra.items():
        if fp[-1] == floor:
            f_c_tra[rp] = fp
    return f_c_tra

@clock
def tst_rss_crd(f_c_tra, f_c_tst, tst_rp):

    # rps get votes
    rp_vote = {}
    # 0 -> VOTE
    # 1 -> VOTE_POINT
    statue = 1
    # 阈值默认值
    threshold = 0

    for rp, fp in f_c_tra.items():
        rp_vote[rp] = 0
        fp_lens = len(fp) - 3
        # 接收不到的信号不计入

        for i in range(fp_lens):
            if float(fp[i]) == -105 and float(f_c_tst[tst_rp][i]) == -105:
                continue
            else:
                fp_diff = abs(float(fp[i]) - float(f_c_tst[tst_rp][i]))

            # 阈值的确定!!!

            # VOTES
            if statue == 0:
                if fp_diff >= threshold:
                    rp_vote[rp] = rp_vote[rp] + 1
            # Vote -> Point 得分量化
            # elif statue == 1:
            #     if fp_diff <= threshold:
            #         rp_vote[rp] = rp_vote[rp] + (threshold - fp_diff) / threshold

            # VOTEP 误差累积
            elif statue == 1:
                if fp_diff >= threshold:
                    rp_vote[rp] = rp_vote[rp] + (fp_diff - threshold)

    # sorted
    rp_vote = OrderedDict(sorted(rp_vote.items(), key=lambda d: d[1]))
    # for rp in rp_vote.keys():
    #     selected_rp = rp
    #     break
    # print(selected_rp)

    # print(rp_vote)
    weight = {}
    # sum = 0
    # sum11 = 0
    for rp, dis in rp_vote.items():
        if dis != 0:
            weight[rp] = 1 / dis
        else:
            weight[rp] = 1
    #     sum = sum + weight[rp]
    # for rp, dis in rp_vote.items():
    #     if dis != 0:
    #         weight[rp] = 1 / dis / sum
    #     else:
    #         weight[rp] = 1 / sum
    weight = OrderedDict(weight)
    # print(weight)

    tag = 1
    xcoor, ycoor, sum_weight = 0, 0, 0

    k = 4

    for rp, w in weight.items():
        if tag > k:
            break
        else:
            tag = tag + 1
            rpx, rpy = float(f_c_tra[rp][-3]), float(f_c_tra[rp][-2])
            xcoor = xcoor + rpx * w
            ycoor = ycoor + rpy * w
            sum_weight = sum_weight + w

    xcoor = float(Decimal(xcoor / sum_weight).quantize(Decimal('0.00')))
    ycoor = float(Decimal(ycoor / sum_weight).quantize(Decimal('0.00')))

    [realx, realy] = f_c_tst[tst_rp][-3:-1]

    vis_temp = [tst_rp, realx, realy, xcoor, ycoor]

    visible_file = r'E:\db\votep613ori_' + '%s%s' % ('th', threshold) + '_3rdv.csv'
    with open(visible_file, 'a+', newline='') as vis:
        writer = csv.writer(vis)
        writer.writerow(vis_temp)

    error_dis = (float(xcoor) - float(realx)) ** 2 + (float(ycoor) - float(realy)) ** 2

    error_dis = float(Decimal(math.sqrt(error_dis)).quantize(Decimal('0.00')))

    # print('%s target coordinates : (%s, %s) , error = %s' % (tst_rp, xcoor, ycoor, error_dis))

    return error_dis

# 参数依次为list,抬头,X轴标签,Y轴标签,XY轴的范围
def draw_plot_month(myList, Title, Xlabel, Ylabel, No):
    y1 = myList
    x1 = range(1, 16, 1)
    plt.figure(int(No))
    plt.plot(x1, y1, label='Error VOTE', linewidth=1, color='r', marker='o', markerfacecolor='green', markersize=2)
    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)
    plt.title(Title)
    plt.savefig('VOTE-Error-75 of month No.%s.png' % No)
    logging.info('Fig saved!')
    plt.legend()
    plt.ion()
    plt.pause(1)
    plt.close()

def select_tra_tst(tl):
    logging.debug('Reading train_data and test_data.')

    filename = "E:\\db\\" + tl[0] + "\\trn" + tl[1] + "rss.csv"
    tst_filename = "E:\\db\\" + tl[2] + "\\tst" + tl[3] + "rss.csv"

    floor = '3'
    fp_coor_tra = floor_filter(rss_crd(filename), floor)
    fp_coor_tst = floor_filter(rss_crd(tst_filename), floor)

    error_s = []
    for rp in range(len(fp_coor_tst)):
        if floor == '5':
            if tl[3] != '04':
                e_dis = tst_rss_crd(fp_coor_tra, fp_coor_tst, 'rp' + str(rp + 48))
            elif tl[3] == '04':
                e_dis = tst_rss_crd(fp_coor_tra, fp_coor_tst, 'rp' + str(rp + 68))
            error_s = error_s + [e_dis]
        elif floor == '3':
            e_dis = tst_rss_crd(fp_coor_tra, fp_coor_tst, 'rp' + str(rp))
            error_s = error_s + [e_dis]

    # 75%的定位误差
    err_75 = np.percentile(np.array(error_s), 75)
    err_75 = float(Decimal(err_75).quantize(Decimal('0.00')))

    output_file = 'E:\\db\\cdf2019\\wknn-md.csv'
    with open(output_file, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(error_s)
    logging.debug('error_list -> .csv file: OK!')

    logging.debug('max error = %s, min error = %s' % (max(error_s), min(error_s)))

    # 画直方图

    # draw_plot_rp(error_s, 'Error Distance Plot', 'RP', 'error/m')  # 直方图展示
    # draw_hist(perimeterList,'perimeterList','Area','number',40.0,80,0.0,8)
    # draw_error_acc(output_file, 'Error CDF Graph', 'error/m', 'percentage')  # 累计误差分布图

    return err_75

def month_error_75_func(month_start, month_end, data_no, tst_no):

    error_list = []
    for month in trange(month_start, month_end + 1):
        error_75 = select_tra_tst(list(map(n2s, [month, data_no, month, tst_no])))
        error_list = error_list + [error_75]
    print(error_list)
    return error_list

def main():
    # 配置日志设置
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)

    logging.info('Start test!')
    output_merge_file = 'E:\\db\\vote_error\\votep613ori_' + '%s%s' % ('th', 11) + '_3rd.csv'
    # 测试序列
    test_m_end = 15
    test_m_start = 1
    test_tra_no = 1
    for test_tst_no in range(1, 6):
        error_list = month_error_75_func(test_m_start, test_m_end, test_tra_no, test_tst_no)
        # draw_plot_month(error_list, 'Error Distance Plot TEST NO.%s' % test_tst_no, 'Month', 'error/m', test_tst_no)  # 直方图展示
        with open(output_merge_file, 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # writer.writerow([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
            writer.writerow(error_list)

    logging.info('Test end!')

if __name__ == "__main__":
    main()
