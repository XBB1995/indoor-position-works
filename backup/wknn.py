import csv
import math
from datetime import datetime
from collections import OrderedDict
from decimal import Decimal
# import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import timeit
import time
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


def rss_crd_row(tra_filename):
    """
    :param tra_filename:
    :return:
    """
    # get raw data from .csv file
    # fp_coor per row

    fp_coor = {}

    with open(tra_filename) as f:
        reader = list(csv.reader(f))
        fp_len = len(reader[0])
        for i in range(len(reader)):
            fp = [None] * fp_len
            if i % 6 == 0:
                fp_coor['row' + str(i)] = fp
            else:
                for j in range(fp_len):
                    if reader[i][j] == '100':
                        fp[j] = -105
                    else:
                        fp[j] = int(reader[i][j])
                fp_coor['row' + str(i)] = fp

    # get crd match fp

    crd_filename = tra_filename.split('.')[0][:-3] + 'crd.csv'
    with open(crd_filename) as crd:
        coor = list(csv.reader(crd))
        crd_len = len(coor)
        for i in range(crd_len):
            if i % 6 != 0:
                fp_coor['row' + str(i)] = fp_coor['row' + str(i)] + coor[i]
    return fp_coor


def floor_filter(tra, floor):
    f_c_tra = {}
    for rp, fp in tra.items():
        if fp[-1] == floor:
            f_c_tra[rp] = fp
    return f_c_tra


def radius_get(f_c_tra):
    """
    :param f_c_tra:
    :return: radius_dict:
    """
    radius_dict = {}
    euclid_dis = {}
    for out_rp in f_c_tra.keys():
        for in_rp, fp in f_c_tra.items():
            fp_lens = len(fp) - 3
            temp_dis = 0
            for i in range(fp_lens):
                temp_dis = temp_dis + (int(fp[i]) - int(f_c_tra[out_rp][i])) ** 2
            euclid_dis[in_rp] = [float(Decimal(math.sqrt(temp_dis)).quantize(Decimal('0.000'))),
                                 [f_c_tra[in_rp][-3], f_c_tra[in_rp][-2]]]
        euclid_dis = OrderedDict(sorted(euclid_dis.items(), key=lambda d: d[1]))
        min_dis = list(euclid_dis.values())
        for i in range(len(min_dis)):
            if min_dis[i][1] != min_dis[0][1]:
                radius_dict[out_rp] = min_dis[i][0]
                break
    return radius_dict

@clock
def tst_rss_crd(f_c_tra, f_c_tst, k, tst_rp, radius_dict={}):
    # euclidean metric
    euclid_dis = {}

    for rp, fp in f_c_tra.items():
        temp_dis = 0
        fp_lens = len(fp) - 3
        for i in range(fp_lens):
            temp_dis = temp_dis + (int(fp[i]) - int(f_c_tst[tst_rp][i])) ** 2

        euclid_dis[rp] = float(Decimal(math.sqrt(temp_dis)).quantize(Decimal('0.00')))

    # using r
    for rp, dis in euclid_dis.items():
        radius = radius_dict[rp]
        euclid_dis[rp] = dis / radius

    # sort
    euclid_dis = OrderedDict(sorted(euclid_dis.items(), key=lambda d: d[1]))
    # print(euclid_dis)

    weight = {}
    for rp, dis in euclid_dis.items():
        weight[rp] = 1 / dis
    weight = OrderedDict(weight)

    tag = 1
    xcoor, ycoor, sum_weight = 0, 0, 0

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

    # visible_file = r'E:\db\wknn919_point_visible.csv'
    # with open(visible_file, 'a+', newline='') as vis:
    #     writer = csv.writer(vis)
    #     writer.writerow(vis_temp)

    error_dis = (xcoor - float(realx)) ** 2 + (ycoor - float(realy)) ** 2

    error_dis = float(Decimal(math.sqrt(error_dis)).quantize(Decimal('0.00')))

    # print('%s target coordinates : (%s, %s) , error = %s' % (tst_rp, xcoor, ycoor, error_dis))

    return error_dis


# 参数依次为list,抬头,X轴标签,Y轴标签,XY轴的范围
def draw_plot_rp(myList, Title, Xlabel, Ylabel):
    y1 = myList
    x1 = range(0, 96, 1)
    plt.plot(x1, y1, label='Error', linewidth=1, color='r', marker='o', markerfacecolor='blue', markersize=3)
    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)
    plt.title(Title)
    plt.legend()
    plt.show()
# 参数依次为list,抬头,X轴标签,Y轴标签,XY轴的范围
def draw_plot_month(myList, Title, Xlabel, Ylabel, No):
    y1 = myList
    x1 = range(1, 16, 1)
    plt.figure(int(No))
    plt.plot(x1, y1, label='Error WKNN', linewidth=1, color='r', marker='o', markerfacecolor='green', markersize=2)
    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)
    plt.title(Title)
    plt.savefig('WKNN-Error-75 of month No.%s.png' % No)
    logging.info('Fig saved!')
    plt.legend()
    plt.ion()
    plt.pause(1)
    plt.close()

def draw_error_acc(error_file, Title, Xlabel, Ylabel):
    lines = []
    with open(error_file, 'r') as f:
        lines = f.read().split('\n')

    dataSets = []
    for line in lines:
        # print(line)
        try:
            dataSets.append(line.split(','))
        except:
            print("Error: Exception Happened... \nPlease Check Your Data Format... ")

    temp = []
    for set in dataSets:
        temp2 = []
        for item in set:
            if item != '':
                temp2.append(float(item))
        temp2.sort()
        temp.append(temp2)
    dataSets = temp

    for set in dataSets:
        plotDataset = [[], []]
        count = len(set)
        for i in range(count):
            plotDataset[0].append(float(set[i]))
            plotDataset[1].append((i + 1) / count)
        # print(plotDataset)
        plt.plot(plotDataset[0], plotDataset[1], '-', linewidth=2)

    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)
    plt.title(Title)
    # plt.savefig(error_file[:-4] + '.png')
    # logging.info('error CDF saved!')
    plt.show()

def select_tra_tst(tl):
    logging.debug('Reading train_data and test_data.')

    filename = "E:\\db\\" + tl[0] + "\\trn" + tl[1] + "rss.csv"
    tst_filename = "E:\\db\\" + tl[2] + "\\tst" + tl[3] + "rss.csv"
    output_file = 'E:\\db\\output\\' + tl[0] + tl[1] + tl[2] + tl[3]
    if os.path.exists(output_file):
        output_file = output_file + '\\output_error.csv'
    else:
        os.makedirs(output_file)
        output_file = output_file + '\\output_error.csv'

    w_k = 1
    floor = '3'

    # fp_coor_tra = floor_filter(rss_crd(filename), floor)
    # fp_coor_tst = floor_filter(rss_crd(tst_filename), floor)
    # radius_d = {}

    # using r
    fp_coor_tra = floor_filter(rss_crd_row(filename), floor)
    radius_d = radius_get(fp_coor_tra)
    fp_coor_tst = floor_filter(rss_crd_row(tst_filename), floor)

    error_s = []
    for rp in range(len(fp_coor_tst)):
        # if floor == '5':
        #     if tl[3] != '04':
        #         e_dis = tst_rss_crd(fp_coor_tra, fp_coor_tst, w_k, 'rp' + str(rp + 48), radius_d)
        #     elif tl[3] == '04':
        #         e_dis = tst_rss_crd(fp_coor_tra, fp_coor_tst, w_k, 'rp' + str(rp + 68), radius_d)
        #     error_s = error_s + [e_dis]
        if floor == '3':
            if rp % 6 != 0:
                e_dis = tst_rss_crd(fp_coor_tra, fp_coor_tst, w_k, 'row' + str(rp), radius_d)
                error_s = error_s + [e_dis]

    # 75%的定位误差
    err_75 = np.percentile(np.array(error_s), 75)
    err_75 = float(Decimal(err_75).quantize(Decimal('0.00')))

    output_file = 'E:\\db\\cdf-r\\k1\\wknn-m2-r.csv'
    with open(output_file, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(error_s)
    logging.debug('error_list -> .csv file: OK!')

    logging.debug('max error = %s, min error = %s' % (max(error_s), min(error_s)))

    # 画直方图
    #draw_plot_rp(error_s, 'Error Distance Plot', 'RP', 'error/m')  # 直方图展示
    #draw_error_acc(output_file, 'Error CDF Graph', 'error/m', 'percentage')  # 累计误差分布图

    return err_75

def month_error_75_func(month_start, month_end, data_no, tst_no):

    error_list = []
    for month in trange(month_start, month_end + 1):
        error_75 = select_tra_tst(list(map(n2s, [month, data_no, month, tst_no])))
        error_list = error_list + [error_75]
    return error_list

def main():
    # 配置日志设置
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)

    logging.info('Start test!')
    # 测试序列
    output_merge_file = 'E:\\db\\cdf2019\\wknn-md-5.13.csv'
    test_m_end = 2
    test_m_start = 2
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
