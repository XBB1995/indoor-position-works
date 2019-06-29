import csv
import math
from datetime import datetime
from collections import OrderedDict
from decimal import Decimal
import numpy as np
from matplotlib import pyplot as plt
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
            fp = [None] * fp_len
            for j in range(fp_len):
                if reader[i][j] == '100':
                    pass
                else:
                    fp[j] = int(reader[i][j])
            fp_coor['row' + str(i)] = fp

    # get crd match fp

    crd_filename = tra_filename.split('.')[0][:-3] + 'crd.csv'
    with open(crd_filename) as crd:
        coor = list(csv.reader(crd))
        crd_len = len(coor)
        for i in range(crd_len):
            fp_coor['row' + str(i)] = fp_coor['row' + str(i)] + coor[i]
    return fp_coor

def floor_filter(tra, floor):
    f_c_tra = {}
    for rp, fp in tra.items():
        if fp[-1] == floor:
            f_c_tra[rp] = fp
    return f_c_tra

def tst_rss_crd(f_c_tra, f_c_tst, sigma, k, tst_rp):
    # euclidean metric
    likelihood = {}

    for rp, fp in f_c_tra.items():
        fp_lens = len(fp) - 3
        temp_list = [math.log(1e-6)] * fp_lens
        for i in range(fp_lens):
            if fp[i] and f_c_tst[tst_rp][i] is not None:
                temp = math.exp(-((int(fp[i]) - int(f_c_tst[tst_rp][i])) ** 2) / (2 * sigma ** 2))
                temp = temp / (math.sqrt(2 * math.pi * sigma ** 2))
                temp_list[i] = math.log(temp)
        likelihood[rp] = sum(temp_list)

    # sorted
    likelihood = OrderedDict(sorted(likelihood.items(), key=lambda d: d[1], reverse=True))

    tag = 1
    xcoor, ycoor, sum_weight = 0, 0, 0

    for rp in likelihood.keys():
        if tag > k:
            break
        else:
            tag = tag + 1
            rpx, rpy = float(f_c_tra[rp][-3]), float(f_c_tra[rp][-2])
            xcoor = xcoor + rpx
            ycoor = ycoor + rpy

    xcoor = float(Decimal(xcoor / k).quantize(Decimal('0.00')))
    ycoor = float(Decimal(ycoor / k).quantize(Decimal('0.00')))

    [realx, realy] = f_c_tst[tst_rp][-3:-1]

    vis_temp = [tst_rp, realx, realy, xcoor, ycoor]

    visible_file = r'E:\db\gk_point_visible.csv'
    with open(visible_file, 'a+', newline='') as vis:
        writer = csv.writer(vis)
        writer.writerow(vis_temp)

    error_dis = (xcoor - float(realx)) ** 2 + (ycoor - float(realy)) ** 2

    error_dis = float(Decimal(math.sqrt(error_dis)).quantize(Decimal('0.00')))

    # print('%s target coordinates : (%s, %s) , error = %s' % (tst_rp, xcoor, ycoor, error_dis))

    return error_dis

# 参数依次为list,抬头,X轴标签,Y轴标签,XY轴的范围
def draw_plot_month(myList, Title, Xlabel, Ylabel, No):
    y1 = myList
    x1 = range(1, 16, 1)
    plt.figure(int(No))
    plt.plot(x1, y1, label='Error GK', linewidth=1, color='r', marker='o', markerfacecolor='green', markersize=2)
    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)
    plt.title(Title)
    plt.savefig('WKNN-Error-75 of month No.%s.png' % No)
    logging.info('Fig saved!')
    plt.legend()
    plt.ion()
    plt.pause(1)
    plt.close()

def select_tra_tst(tl):
    logging.debug('Reading train_data and test_data.')

    filename = "E:\\db\\" + tl[0] + "\\trn" + tl[1] + "rss.csv"
    tst_filename = "E:\\db\\" + tl[2] + "\\tst" + tl[3] + "rss.csv"
    # output_file = 'E:\\db\\output\\' + tl[0] + tl[1] + tl[2] + tl[3]
    # if os.path.exists(output_file):
    #     output_file = output_file + '\\output_error.csv'
    # else:
    #     os.makedirs(output_file)
    #     output_file = output_file + '\\output_error.csv'

    w_k = 6
    sigma  = 4
    floor = '3'
    fp_coor_tra = floor_filter(rss_crd(filename), floor)
    fp_coor_tst = floor_filter(rss_crd(tst_filename), floor)

    error_s = []
    for rp in range(len(fp_coor_tst)):
        if floor == '5':
            if tl[3] != '04':
                e_dis = tst_rss_crd(fp_coor_tra, fp_coor_tst, sigma, w_k, 'row' + str(rp + 48 * 6))
            elif tl[3] == '04':
                e_dis = tst_rss_crd(fp_coor_tra, fp_coor_tst, sigma, w_k, 'row' + str(rp + 68 * 6))
            error_s = error_s + [e_dis]
        elif floor == '3':
            e_dis = tst_rss_crd(fp_coor_tra, fp_coor_tst, sigma, w_k, 'row' + str(rp))
            error_s = error_s + [e_dis]

    # 75%的定位误差
    err_75 = np.percentile(np.array(error_s), 75)
    err_75 = float(Decimal(err_75).quantize(Decimal('0.00')))

    output_file = 'E:\\db\\make_cdf\\gk.csv'
    with open(output_file, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(error_s)
    # logging.debug('error_list -> .csv file: OK!')
    #
    # logging.debug('max error = %s, min error = %s' % (max(error_s), min(error_s)))

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
    output_merge_file = 'E:\\db\\gk_error\\error_of_tst.csv'
    test_m_end = 15
    test_m_start = 1
    test_tra_no = 1
    for test_tst_no in range(1, 6):
        error_list = month_error_75_func(test_m_start, test_m_end, test_tra_no, test_tst_no)
        draw_plot_month(error_list, 'Error Distance Plot TEST NO.%s' % test_tst_no, 'Month', 'error/m', test_tst_no)  # 直方图展示
        with open(output_merge_file, 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # writer.writerow([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
            writer.writerow(error_list)
    logging.info('Test end!')


if __name__ == "__main__":
    main()
