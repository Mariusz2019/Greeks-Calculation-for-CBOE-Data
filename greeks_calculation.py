#SNIPPET TO CALCULATE OPTIONS IV AND GREEKS FROM OPTION CHAIN
#THERE IS NO NEED TO GET INTEREST RATE AND DIVIDEND YIELD
#BASED ON: EXCELLENT WORK BY ADAM LEE SPEIGHT - https://github.com/aspeight/spx_data_munging (IV CALCULATION)
#BASED ON: OPTIONPY (GREEKS CALCULATION) - https://github.com/ibaris/optionpy/tree/main/optionpy
#MAIN PURPOSE: GREEK CALCULATION FOR CBOE RAW DATA

#V.21 - OWN CALCULATION GREEKS BASED ON OPTIONPY BUT ALL IN POLARS

import numpy as np
import scipy.misc
import cvxpy as cp
import polars as pl


pl.Config(
    set_fmt_float="full",
    tbl_cell_numeric_alignment="RIGHT",
    thousands_separator=",",
    tbl_width_chars=200,
    fmt_str_lengths=1000,
    set_tbl_cols=-1,
    set_tbl_rows=-1,
    set_tbl_hide_column_data_types=True,
    set_tbl_hide_dataframe_shape=True

)

_gaussian = scipy.stats.norm()
_N_cdf = _gaussian.cdf
_N_pdf = _gaussian.pdf
_N_invcdf = _gaussian.ppf

_DAYS_PER_YEAR = 365.

# ADAM LEE SPEIGHT'S FUNCTIONS
def fwd_greeks_from(fwd, strike, tau, sigma, is_call, ref_tau=1./12.):
    '''Computes Black-Scholes prices and greeks assuming zero interest and dividend.

    Inputs are assumed to be np.array or pandas.Series.

    Returns a dictionary

    See
    https://en.wikipedia.org/wiki/Greeks_(finance)
    '''
    sqrt_tau = np.sqrt(tau)
    log_fwd_over_strike = np.log(fwd/strike)
    d1 = (log_fwd_over_strike + 0.5*sigma**2 * tau) / (sigma*sqrt_tau)
    d2 = d1 - sqrt_tau * sigma

    Nd1 = _N_cdf(d1)
    phid1 = _N_pdf(d1)
    Nd2 = _N_cdf(d2)
    Nmd1 = _N_cdf(-d1)
    Nmd2 = _N_cdf(-d2)

    call_price = fwd * Nd1 - strike * Nd2
    put_price = -fwd * Nmd1 + strike * Nmd2

    call_delta = Nd1
    put_delta = -Nmd1

    # Note: in this special case, theta is the same for call and put options
    theta = -0.5*fwd * (sigma/sqrt_tau) * phid1
    gamma = (phid1) / (fwd * sigma * sqrt_tau)
    speed = -(gamma / fwd) * (1. + d1 / (sigma * sqrt_tau))

    vega = fwd * sqrt_tau * phid1
    wt_vega = np.sqrt(ref_tau / tau) * vega # todo: check correctness
    vanna = (vega/fwd)*(1-d1/(sigma*sqrt_tau))
    vomma = vega * d1 * d2 / sigma
    ultima = (-vega/sigma**2)*(d1*d2*(1-d1*d2)+d1**2+d2**2)

    result = dict(  call_price=call_price,
                    put_price=put_price,
                    call_delta=call_delta,
                    put_delta=put_delta,
                    theta=theta,
                    gamma=gamma,
                    speed=speed,
                    vega=vega,
                    vanna=vanna,
                    vomma=vomma,
                    ultima=ultima,
                    wt_vega=wt_vega,
                )

    # todo: fill nan with zero before multiplying by bool
    result['price'] = 1.*is_call * call_price + (1.-is_call)*put_price
    result['delta'] = 1.*is_call * call_delta + (1.-is_call)*put_delta
    result['abs_delta'] = np.abs(result['delta'])

    return result

def refine_iv(tgt, price, vega, vomma, ultima, order=3):

    '''One iteration of Newton-like method for implied vol calculation

    A higher order generalization of Newton's method is supported.
    See https://en.wikipedia.org/wiki/Householder%27s_method

    Params
    ------
    tgt : (np.array) observed option price (calibration target)
    price : (np.array) model-computed price (call or put) given sigma
    vega, vomma, ultima : (np.array) model-computed greeks given sigma
    order : (int) 1=Newton's method, 2-3 are higher order Householder methods

    Returns
    -------
    An array (compatible with tgt) that, when added to the current
    implied volatility, gives an improved estimate. That is,
    iv -> iv + update.

    Notes
    -----
    The paper by Li (2006) provides a useful domain for when this
    type of iteration can be expected to converge:
    |x| <= 0.5, 0 << v <= 1, and |x/v| <= 2,
    where x = log(F/K), F=exp((r-q)*tau)*spot, and v = sqrt(tau)*sigma.

    Generally, starting with a sigma near the upper end of Li's domain
    gives good convergence rates.
    '''
    x = tgt - price
    h = x / vega
    if order==1:
        update = h
    elif order==2:
        update = h * (1./(1 + 0.5*(h)*(vomma/vega)))
    elif order==3:
        update = (h
                  * (1 + 0.5*(vomma/vega)*h)
                  / (1 + (vomma/vega)*h + (1./6.)*(ultima/vega)*h**2 ))
    else:
        raise ValueError("order must be 1,2 or 3, not {}".format(order))
    return update

def fwd_raw_compute_iv(tgt, fwd, strike, tau, is_call,
                       initial_sigma=2., num_iters=12, order=3):
    '''Apply Newton-like iteration to solve for implied vol with no error checks.

    '''
    sigma = initial_sigma * (1. + 0*tgt)
    for it in range(num_iters):
        greeks = fwd_greeks_from(fwd=fwd, strike=strike, tau=tau, sigma=sigma, is_call=is_call)
        update = refine_iv(tgt=tgt,
                           price=greeks['price'],
                           vega=greeks['vega'],
                           vomma=greeks['vomma'],
                           ultima=greeks['ultima'],
                           order=order)
        sigma += update
    return sigma

def fwd_safe_compute_iv(tgt, fwd, strike, tau, is_call,
                        initial_sigma=1.5,
                        num_iters=12,
                        order=3,
                        sigma_bounds=(0.01,2.),
                        price_tol=None):
    '''Apply Newton-like iteration to solve for implied vol with some error checking'''
    greeks_low = fwd_greeks_from(fwd=fwd, strike=strike, tau=tau, sigma=sigma_bounds[0], is_call=is_call)
    greeks_high = fwd_greeks_from(fwd=fwd, strike=strike, tau=tau, sigma=sigma_bounds[1], is_call=is_call)

    clip_tgt = np.clip(tgt, greeks_low['price'], greeks_high['price'])

    iv = fwd_raw_compute_iv(clip_tgt, fwd, strike, tau, is_call,
                            initial_sigma=initial_sigma,
                            num_iters=num_iters, order=order)

    greeks = fwd_greeks_from(fwd=fwd, strike=strike, tau=tau, sigma=iv, is_call=is_call)
    #iv[clip_tgt != tgt] = np.nan # todo: float equality check is sometimes not what we want
    if price_tol is not None:
        iv[np.abs(greeks['price']-tgt)>price_tol] = np.nan

    return iv


def spot_compute_iv(tgt, spot, strike, tau, is_call, int_rate, div_yld):
    fwd = spot * np.exp(tau * (int_rate - div_yld))
    fwd_tgt = tgt * np.exp(tau * int_rate)
    return fwd_safe_compute_iv(tgt=fwd_tgt, fwd=fwd, strike=strike, tau=tau, is_call=is_call)


def solve_marks(chain, S0, dte, obj_weight=0., int_rate=None, div_yld=None):
    """[summary]

    Args:
        chain (DataFrame): Indexed by strike prices, columns are multi-index
            with mid,spread at first level and CALL,PUT at second
        S0 (float): An approximate value of the spot price
        dte (int): Days to expiration for the option chain
        obj_weight (float, optional):  Mix between objective functions:
            0 -> sum of absolute deviation, 1 -> sum of squared deviation.
        int_rate ([float,None], optional): If specified, constrains the discount factor.
        div_yld ([float,None], optional): If specified, constrains the forward price

    Returns:
        dict: Keys include forward price (F), discount factor (D), int rate (r),
              and a dataframe (marks) with columns for call and put marks
              as well as the deviation from mid price as percentage of bid/ask spread.
    """
    tau = dte / 365.
    num_strikes = chain.shape[0]
    psi = cp.Variable(name='psi')
    delta = cp.Variable(name='delta')
    eps_call = cp.Variable(num_strikes, name='eps_call')
    eps_put = cp.Variable(num_strikes, name='eps_put')

    S = S0*(1-psi)
    D = 1-delta
    K = chain['strike'].to_numpy()

    m_call = chain['mid_C'].to_numpy()
    m_put = chain['mid_P'].to_numpy()

    spr_call = chain['spread_C'].to_numpy()
    spr_put = chain['spread_P'].to_numpy()

    C = m_call + cp.multiply(spr_call, eps_call)
    P = m_put + cp.multiply(spr_put, eps_put)

    objective = (
        obj_weight * (cp.sum_squares(eps_call) + cp.sum_squares(eps_put))
        + (1-obj_weight) * (cp.sum(cp.abs(eps_call)) + cp.sum(cp.abs(eps_put)))
    )

    constraints = [C - P - (S - D*K) == 0]
    if int_rate is not None:
        constraints.append(D == np.exp(-int_rate*tau))
    if div_yld is not None:
        assert int_rate is not None
        constraints.append(S == S0*np.exp(-div_yld*tau))

    prob = cp.Problem(
        cp.Minimize(objective),
        constraints,
    )
    prob.solve()

    marks = (pl.DataFrame({
                'eps_call': np.array(eps_call.value),
                'eps_put': np.array(eps_put.value),
                'mark_call': np.array(C.value),
                'mark_put': np.array(P.value),
                'strike': chain['strike']
                }
                )
                .with_columns(
                    diff_call=pl.col('eps_call') * pl.lit(spr_call),
                    diff_put=pl.col('eps_put') * pl.lit(spr_put),
                )
    )

    return dict(
        delta=delta.value,
        psi=psi.value,
        S=S.value,
        F=S.value / D.value,
        D=D.value,
        r=np.log(D.value)/(-tau),
        marks=marks,
    )


#read one of sample files from CBOE shop - to be downloades from its shop
filename = 'cboe/UnderlyingOptionsIntervals_1800sec_calcs_oi_2021-04-26.csv'


#read all columns using polars
option_data = (pl
                .scan_csv(filename)
                .filter(pl.col('underlying_symbol') == "A")
                .with_columns(
                    dte=(pl.col("expiration").str.strptime(pl.Datetime, "%Y-%m-%d") - pl.col("quote_datetime").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")).dt.total_days(),
                    is_call=(pl.col('option_type') == 'C'),
                    mid=(pl.lit(0.5) * (pl.col('bid') + pl.col('ask'))),
                    spread=(pl.col('ask') - pl.col('bid'))
                )
                .with_columns(
                    tau=(pl.col('dte') / pl.lit(_DAYS_PER_YEAR))
                    )
                .collect()
               )

snapshots = sorted(option_data['quote_datetime'].unique())
S0 = 900

'''code to calculate interest rate, dividend yield and IV also taken from Adam Lee Speight examples, modified for the use of polars library'''
snapshot_list = []
for snapshot in snapshots:
    sel_option_data = (option_data
                       .filter(pl.col('quote_datetime') == snapshot)
                       )
    dtes = sorted(sel_option_data['dte'].unique())

    chains = (sel_option_data
              .select(['strike', 'dte', 'option_type', 'bid', 'ask', 'is_call', 'tau', 'mid', 'spread'])
              )

    marks_by_dte = {}
    for dte in dtes:
        chain = (chains
                 .filter((pl.col('dte') == dte) & pl.col('dte').is_not_nan() & pl.col('bid').is_not_nan() & pl.col('ask').is_not_nan())
                 .pivot(index='strike', on='option_type', values=['dte', 'bid', 'ask', 'mid', 'spread'], aggregate_function=None).sort('strike')
                 )


        marks_by_dte[dte] = solve_marks(chain, S0=S0, dte=dte, obj_weight=0.)


    rates = pl.DataFrame([
        (dte, dte / _DAYS_PER_YEAR, data['r'], data['F'])
        for dte, data in marks_by_dte.items()
    ], orient="row")
    rates.columns = ['dte', 'tau', 'int_rate', 'fwd']
    rates = (rates
             .with_columns(
                spot=pl.col('fwd').first() # use front month fwd as proxy for spot
                )
             )

    yieldd = [np.log(rates.filter(pl.col('dte') == dte)['spot'][0] / data['F']) / (rates.filter(pl.col('dte') == dte)['tau'][0]) + data['r']
                        for dte, data in marks_by_dte.items()]

    rates = (rates
             .with_columns(
                div_yld=pl.Series(yieldd)
            )
    )

    marks_list = []
    for dte, data in marks_by_dte.items():
        marks = (data['marks'][['strike', 'mark_call', 'mark_put']]
                 .clone()
                 .rename({'mark_call':'C', 'mark_put':'P'})
                 .unpivot(index='strike', on=['C', 'P'])
                 .sort('strike')
                 .with_columns(dte=pl.lit(dte))
                 .rename({'variable':'option_type', 'value':'mark'})
                 )
        marks_list.append(marks)

    clean_marks = (pl
                    .concat(marks_list)
                    .with_columns(dte=pl.col('dte').cast(pl.Int64))
                    .join(rates[['dte', 'tau', 'int_rate', 'div_yld', 'spot']],
                        on='dte', how='left')
                   .with_columns(
                        is_call=pl.when(pl.col('option_type') == 'C').then(pl.lit(True)).otherwise(pl.lit(False)),
                        flag=pl.when(pl.col('option_type') == 'C').then(pl.lit(1)).otherwise(pl.lit(-1))
                    )
                )

    imp_vol = spot_compute_iv(
                            tgt=clean_marks['mark'],
                            spot=clean_marks['spot'],
                            strike=clean_marks['strike'],
                            tau=clean_marks['tau'],
                            is_call=clean_marks['is_call'],
                            int_rate=clean_marks['int_rate'],
                            div_yld=clean_marks['div_yld'],
                            )

    #compute own greeks based on optionpy
    clean_marks = (clean_marks
                    .with_columns(
                        imp_vol=pl.Series(imp_vol)
                    )
                    .with_columns(
                        d1=(
                                   np.log(pl.col('spot') / pl.col('strike')) +
                                   (pl.col('int_rate') - pl.col('div_yld') + pl.lit(.5) * pl.col('imp_vol').pow(2)) * pl.col('tau')
                           ) /
                           (
                                   (pl.col('imp_vol') * np.sqrt(pl.col('tau'))) + pl.lit(1e-6)
                           )
                        )
                    .with_columns(
                        d2=(
                                pl.col('d1') - ((pl.col('imp_vol') * np.sqrt(pl.col('tau'))) + pl.lit(1e-6))
                            )
                        )
                    .with_columns(
                        nd1=scipy.special.ndtr(pl.col('flag') * pl.col('d1')),
                        nd2=scipy.special.ndtr(pl.col('flag') * pl.col('d2')),
                        phi_d1=scipy.stats.norm._pdf(pl.col('d1'))
                        )
                    .with_columns(
                        price=pl.col('flag') * pl.col('spot') * np.exp(-pl.col('div_yld') * pl.col('tau')) * pl.col('nd1') -
                              pl.col('flag') * pl.col('strike') * np.exp(-pl.col('int_rate') * pl.col('tau')) * pl.col('nd2'),
                        delta=pl.col('flag') * np.exp(-pl.col('div_yld') * pl.col('tau')) * pl.col('nd1'),
                        vega=pl.col('spot') * np.exp(-pl.col('div_yld') * pl.col('tau')) * scipy.stats.norm._pdf(pl.col('d1')) * np.sqrt(pl.col('tau')) * pl.lit(0.01),
                        gamma=np.exp(-1 * pl.col('div_yld') * pl.col('tau')) * pl.col('phi_d1') / (pl.col('spot') * pl.col('imp_vol') * np.sqrt(pl.col('tau'))),
                        theta=(
                                -np.exp(-pl.col('div_yld') * pl.col('tau')) * (pl.col('spot') * pl.col('phi_d1') * pl.col('imp_vol')) / (pl.lit(2) * np.sqrt(pl.col('tau'))) +
                                -pl.col('flag') * pl.col('int_rate') * pl.col('strike') * np.exp(-pl.col('int_rate') * pl.col('tau')) * pl.col('nd2') +
                                -pl.col('flag') * pl.col('div_yld') * pl.col('spot') * np.exp(-pl.col('div_yld') * pl.col('tau')) * pl.col('nd1')
                                ) / pl.lit(365),
                     )
                    .select(pl.all().exclude('d1', 'd2', 'nd1', 'nd2', 'phi_d1'))
                    # per ALS adjust for the 100 multiplier
                    # per ALS note that vega needs no adjustment
                    .with_columns(
                        wt_vega=(pl.lit(1./12.) / pl.col('tau')).sqrt() * pl.col('vega'),
                        value=pl.lit(100) * pl.col('price'),
                        theta=pl.lit(100) * pl.col('theta'),
                        delta=pl.lit(100) * pl.col('delta'),
                        quote_datetime=pl.lit(snapshot)
                    )
    )

    snapshot_list.append(clean_marks)

final_data = pl.concat(snapshot_list)

print(final_data
        .filter(
                (pl.col('quote_datetime') == '2021-04-26 16:00:00') &
                (pl.col('strike') == 130)
            )
)


