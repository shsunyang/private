# from kuanke.user_space_api import *
import time
import math
from sklearn.svm import SVR  
import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from jqlib.alpha101 import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from jqlib.technical_analysis import *

class g:
    stockpool='000985.XSHG'

def Filter_paused_stock(stock_list,date):
    return [stock for stock in stock_list if not 
            get_price(stock, count = 1, end_date=date.strftime('%Y-%m-%d'), 
            frequency='daily', fields=['paused'])['paused'][0]]

def Filter_low_limit_stock(stock_list,date):
    stocks=[]
    for stock in stock_list:
        stockprice = get_price(stock, count = 1, 
            end_date=date.strftime('%Y-%m-%d'), 
            frequency='daily', fields=['high_limit','low_limit','open','close'])
        if stockprice['high_limit'][0]==stockprice['close'][0] and stockprice['high_limit'][0]>stockprice['low_limit'][0]:
            stocks.append(stock)
    return stocks

def Filter_high_limit_stock(stock_list,date):
    stocks=[]
    for stock in stock_list:
        stockprice = get_price(stock, count = 1, 
            end_date=date.strftime('%Y-%m-%d'), 
            frequency='daily', fields=['high_limit','low_limit','open','close'])
        if stockprice['low_limit'][0]==stockprice['close'][0] and stockprice['high_limit'][0]>stockprice['low_limit'][0]:
            stocks.append(stock)
    return stocks

def cal_stock_share_inc(stockidx,current_dt):
    stockprice = get_price(stockidx, count = 2, 
        end_date=(current_dt), 
        frequency='daily', fields=['high_limit','low_limit','open','close'])
    inc = 0.0    
    if(stockprice['close'][0]!=0):
        inc = ((stockprice['close'][1]-stockprice['close'][0])/stockprice['close'][0])
    return inc

def cal_stock_share_n_inc(stockidx,current_dt,nday):
    stockprice = get_price(stockidx, count = nday+1, 
        end_date=(current_dt), 
        frequency='daily', fields=['high_limit','low_limit','open','close'])
    inc = 0.0    
    if(stockprice['close'][0]!=0):
        inc = ((stockprice['close'][nday]-stockprice['close'][0])/stockprice['close'][0])
    return inc

def query_security_price(security, start_date, end_date, field='close'):
    h = get_price(
            security=security,
            start_date=start_date,
            end_date=end_date,
            frequency='daily',
            fields=field,
            skip_paused=False,
            fq='pre')[field]
    h.fillna(0)
    df = pd.DataFrame(columns=['code', 'name', 'inc', 'max', 'min'])
    for code in security:
        info = get_security_info(code)
        priceBenchmark = h[code][0]
        priceLatest = h[code][-1]
        priceHighest = h[code].max()
        priceLowest = h[code].min()
        if np.isnan(priceBenchmark):
            continue 
        diff1 = round(100.0*(priceLatest / priceBenchmark - 1.0), 2)
        diff2 = round(100.0*(priceHighest / priceBenchmark - 1.0), 2)
        diff3 = round(100.0*(priceLowest / priceBenchmark - 1.0), 2)
        dfTmp = pd.DataFrame({'code': [code],
                            'name': [info.display_name],
                            'inc': [diff1],
                            'max': [diff2],
                            'min': [diff3]},
                            index=[code])
        #print dfTmp
        df = pd.concat([df, dfTmp])
    df.index = df.code.values
    return df

def draw_figure(Y1,factor,Y2,Y3,current_dt):
    high_limit = Filter_high_limit_stock(factor.index, date = current_dt)
    low_limit = Filter_low_limit_stock(factor.index, date = current_dt)
    #print high_limit
    fig = plt.figure(figsize=(40,160))
    ax = fig.add_subplot(311)#, projection='3d')
    #plt.plot(range(0,11), range(0,11), color='blue', lw=1)
    #plt.plot(range(0,12,2), [0,0,0,0,0,0], color='blue', lw=1)
    #plt.scatter(svr.predict(X), Y, color='darkorange')
    lw = 0.1
    #plt.plot(svr.predict(X), Y, color='navy', lw=lw)
    ax.scatter(Y1, factor, color='yellow', lw=lw)
    #plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
    #plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
    ax.set_ylabel('评估')
    ax.set_xlabel('资产')
    #ax.set_zlabel('资产')
    bx = fig.add_subplot(312)
    bx.scatter(Y2, factor, color='yellow', lw=lw)
    bx.set_ylabel('评估')
    bx.set_xlabel('资产')
    
    cx = fig.add_subplot(313)
    cx.scatter(Y3, factor, color='yellow', lw=lw)
    cx.set_ylabel('评估')
    cx.set_xlabel('资产')

    plt.title(current_dt.strftime('%Y-%m-%d')+
              '涨停比例 '+str(len(high_limit)*100.00/len(factor.index))+
             ' 跌停比例 '+str(len(low_limit)*100.00/len(factor.index)),size = 30)
    plt.legend()
    #x = range(len(X.index),50)
    n = 0
    for idx in factor.index:
        inc = cal_stock_share_inc(idx,current_dt)
        #print(idx,inc)
        #stockprice['close'][0]
        if(inc>=0):
            colortag = 'r'
        else:
            colortag = 'g'
        text = idx[0:6] + '('+ ('%.2f' %Y1[idx])+','+ ('%.2f' %factor[idx])+','+ ('%.2f' %Y1[idx])+')'
        
        if(idx in high_limit or idx in low_limit):
            textsize = 20
            weight = "bold"
            alpha = 1
            rotation=(inc*10*90)
        else:
            textsize = 12
            weight = "light"
            alpha = 0.5
            rotation=(inc*10*90)

        ax.text(Y1[idx]+0.03, factor[idx],# Z[idx],
             text, size = textsize, weight = weight, style = "italic",color = colortag,
             alpha = alpha,rotation=rotation)

        bx.text(Y2[idx], factor[idx],# Z[idx],
             text, size = textsize, weight = weight, style = "italic",color = colortag,
             alpha = alpha,rotation=rotation)

        cx.text(Y3[idx], factor[idx],# Z[idx],
             text, size = textsize, weight = weight, style = "italic",color = colortag,
             alpha = alpha,rotation=rotation)
    #plt.xticks(x, X.index.values, rotation=90)
    #plt.margins(0.08)
    #plt.subplots_adjust(bottom=0.15)
    plt.show()
    return
 
def svr_value(current_dt):
    g.stockpool = '000985.XSHG'
    sample = get_index_stocks(g.stockpool, date = current_dt)
    cxstock=[]
    #cxstock = get_concept_stocks(g.cxpool, date = current_dt)
    #print cxstock
    #sample = Filter_paused_stock(sample,current_dt)

    q = query(valuation.code, 
                valuation.market_cap, 
                balance.total_assets - balance.total_liability,
                balance.total_assets / balance.total_liability,
                income.net_profit, income.net_profit+1,#income.np_parent_company_owners, 
                indicator.inc_revenue_year_on_year,#inc_operation_profit_year_on_year,#inc_revenue_year_on_year,
                cash_flow.net_operate_cash_flow,
                balance.development_expenditure,
                 (indicator.adjusted_profit/balance.total_owner_equities)*100,
                indicator.gross_profit_margin).filter(
                valuation.code.in_(sample) | valuation.code.in_(cxstock),
                #cash_flow.net_operate_cash_flow > 0,
                balance.pubDate<=current_dt,
                income.pubDate<=current_dt)

    df = get_fundamentals(q, date = current_dt)
    #print df
    df.columns = ['code',
        'log_mcap', 
        'log_NC', 'LEV',#'log_REC',
        'NI_p', 'NI_n',
        'g','cash',
        'log_RD','net_rate','gross']
     
    #CYEL2,CYES2 = CYE(list(df['code']),check_date=current_dt)
    #print len(CYES2)
    #arrCye = pd.DataFrame(list(CYES2.items()), columns=['code', 'cye'])
    #arrCye.fillna(0)
    #df['cye']=arrCye['cye']
    df['mcap'] = df['log_mcap']
    df['log_mcap'] = np.log(df['log_mcap'])
    #df['ps_ratio'] = np.log(df['ps_ratio'])
    df['log_NC'] = np.log(df['log_NC'])
    df['NI_p'] = np.log(np.abs(df['NI_p']))

    df['NI_n'] = np.log(np.abs(df['NI_n'][df['NI_n']<0]))
    #df['neg_cash'] =(np.abs(df['cash'][df['cash']<=0]))
    df['cash'] =(np.log(df['cash'][df['cash']>0]))
    #df['g'] =  (np.log(df['g'][df['g']>0])) 
    df['gross'] =  (np.log(df['gross'][df['gross']>0])) 
    df['log_RD'] = np.log(df['log_RD'])
    df.index = df.code.values
    del df['code']
    #df['ninc']=list(map(lambda x:cal_stock_share_n_inc(x,current_dt,250),df.index))
 
    #df['log_REC'] = np.log(df['log_REC'])
    #a01 = alpha_001(current_dt,'all')

    #a01 = a01.fillna(0)

    df = df.fillna(0)
    df[df>100000] = 100000
    df[df<-100000] = -100000
    #print df
    #pd.merge(df,a01)
    #df['a01']=0
    #df[['a01']] = a01



    industry_set = ['801010', '801020', '801030', '801040', '801050', '801080', 
                    '801110', '801120', '801130', '801140', '801150', '801160',
                    '801170', '801180', '801200', '801210', '801230', '801710',
                    '801720', '801730', '801740', '801750', '801760', '801770',
                    '801780', '801790', '801880','801890']
    industry_set_3 = [
        '850111','850112','850113','850121','850122','850131','850141',
        '850151','850152','850154','850161','850171','850181','850211',
        '850221','850222','850231','850241','850242','850311','850313',
        '850321','850322','850323','850324','850331','850332','850333',
        '850334','850335','850336','850337','850338','850339','850341',
        '850342','850343','850344','850345','850351','850352','850353',
        '850361','850362','850363','850372','850373','850381','850382',
        '850383','850411','850412','850521','850522','850523','850531',
        '850541','850542','850543','850544','850551','850552','850553',
        '850611','850612','850614','850615','850616','850623','850711',
        '850712','850713','850714','850715','850716','850721','850722',
        '850723','850724','850725','850726','850727','850728','850729',
        '850731','850741','850751','850811','850812','850813','850822',
        '850823','850831','850833','850841','850851','850852','850911',
        '850912','850913','850921','850935','850936','850941','851012',
        '851013','851014','851021','851111','851112','851113','851114',
        '851115','851121','851122','851231','851232','851233','851234',
        '851235','851236','851241','851242','851243','851244','851311',
        '851312','851313','851314','851315','851316','851322','851323',
        '851324','851325','851326','851327','851411','851421','851432',
        '851433','851434','851435','851441','851511','851512','851521',
        '851531','851541','851551','851561','851611','851612','851613',
        '851614','851615','851621','851631','851641','851711','851721',
        '851731','851741','851751','851761','851771','851781','851811',
        '851821','851911','851921','851931','851941','852021','852031',
        '852032','852033','852033','852041','852051','852052','852111',
        '852112','852121','852131','852141','852151','852211','852221',
        '852222','852223','852225','852226','852241','852242','852243',
        '852244','852311','857221','857231','857232','857233','857234',
        '857235','857241','857242','857243','857244','857251','857321',
        '857322','857323','857331','857332','857333','857334','857335',
        '857336','857341','857342','857343','857344','857411','857421',
        '857431','858811'
    ]
    industry_series = industry_set

    for i in range(len(industry_series)):
        industry = get_industry_stocks(industry_series[i], date = None)
        s = pd.Series([0]*len(df), index=df.index)
        s[set(industry) & set(df.index)]=1
        df[industry_series[i]] = s

    X = df[[#'log_mcap',
        'log_NC', 'LEV',#'ps_ratio','pcf_ratio','turnover_ratio',# 'log_REC',
        'NI_p', 'NI_n', #'neg_cash',
        'g','log_RD',#'gross','cash',#'cye',#'net_rate',#'ninc',#'a01',
        '801010', '801020', '801030', '801040', '801050', '801080', 
        '801110', '801120', '801130', '801140', '801150', '801160',
        '801170', '801180', '801200', '801210', '801230', '801710',
        '801720', '801730', '801740', '801750', '801760', '801770',
        '801780', '801790', '801880','801890'
        ]]
    Y = df[['log_mcap']]#[['log_mcap']]
    X = X.fillna(0)
    Y = Y.fillna(0)

    svr = SVR(kernel='rbf', gamma=0.1) 
    model = svr.fit(X, Y)
    factor = Y - pd.DataFrame(svr.predict(X), index = Y.index,
        columns = ['log_mcap'])
    draw_figure(X['NI_p'],factor['log_mcap'],X['LEV'],X['log_NC'],current_dt)

    factor = factor.sort_values(by = 'log_mcap',ascending=True)
    factor['code'] = list(factor.index)
    X['code'] = factor['code']
    X['predict'] = factor['log_mcap']
    X['mcap'] = df['mcap']
    X['cash'] = df['cash']
    #X['neg_cash'] = df['neg_cash']
    X['net_rate'] = df['net_rate']
    X['gross'] = df['gross']
    X['date'] = current_dt
    #X['cye'] = df['cye']
    return X.sort_values(by = 'predict',ascending=True)

class context:
    current_dt= datetime.now()#datetime(2018,9,20)#
    previous_date=current_dt-timedelta(days=1)
    start_date=datetime(2018,2,1)

factor = svr_value(context.current_dt)
pd.set_option('display.max_columns',16)
pd.set_option('display.max_rows',500)
#factor[factor['neg_cash']==0].sort_values(by='log_RD',ascending=False)
factor
