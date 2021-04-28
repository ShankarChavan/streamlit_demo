import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from decimal import Decimal



def round_num(n, decimals):
    '''
    Params:
    n - number to round
    decimals - number of decimal places to round to
    Round number to 2 decimal places
    For example:
    10.0 -> 10
    10.222 -> 10.22
    '''
    return n.to_integral() if n == n.to_integral() else round(n.normalize(), decimals)

def drop_zero(n):
    '''
    Drop trailing 0s
    For example:
    10.100 -> 10.1
    '''
    n = str(n)
    return n.rstrip('0').rstrip('.') if '.' in n else n

def numerize(n, decimals=2):
    '''
    Params:
    n - number to be numerized
    decimals - number of decimal places to round to
    Converts numbers like:
    1,000 -> 1K
    1,000,000 -> 1M
    1,000,000,000 -> 1B
    1,000,000,000,000 -> 1T
    '''
    is_negative_string = ""
    if n < 0:
        is_negative_string = "-"
    n = abs(Decimal(n))
    if n < 1000:
        return is_negative_string + str(drop_zero(round_num(n, decimals)))
    elif n >= 1000 and n < 1000000:
        if n % 1000 == 0:
            return is_negative_string + str(int(n / 1000)) + "K"
        else:
            n = n / 1000
            return is_negative_string + str(drop_zero(round_num(n, decimals))) + "K"
    elif n >= 1000000 and n < 1000000000:
        if n % 1000000 == 0:
            return is_negative_string + str(int(n / 1000000)) + "M"
        else:
            n = n / 1000000
            return is_negative_string + str(drop_zero(round_num(n, decimals))) + "M"
    elif n >= 1000000000 and n < 1000000000000:
        if n % 1000000000 == 0:
            return is_negative_string + str(int(n / 1000000000)) + "B"
        else:
            n = n / 1000000000
            return is_negative_string + str(drop_zero(round_num(n, decimals))) + "B"
    elif n >= 1000000000000 and n < 1000000000000000:
        if n % 1000000000000 == 0:
            return is_negative_string + str(int(n / 1000000000000)) + "T"
        else:
            n = n / 1000000000000
            return is_negative_string + str(drop_zero(round_num(n, decimals))) + "T"
    else:
        return is_negative_string + str(n)

def calc_years_month(yrs):
    MONTHS=12
    return MONTHS*yrs

@st.cache
def generate_data(n_years):


    np.random.seed(1)
    # Prepare hypothetical data for 5 years i.e. 12 months    
    tot_months=calc_years_month(n_years)
    tot_months=60
    
    subscription_data = {"Month": range(1, int(tot_months)+1), 
                         "Net Growth (%)": np.random.normal(0, 0.05, int(tot_months))}

    return subscription_data,tot_months

def run_simulation(subscription_data,n_rounds,n_years,n_users,r_per_usr):
    # Specify number of monte carlo simulations
    
    results = []

    historical_data = subscription_data["Net Growth (%)"]
    tot_mon=calc_years_month(n_years)

    for rnd in range(n_rounds):
        # Randomy selected growth rates from 12 months of history
        idx = np.random.randint(0,tot_mon , 12)
        selected_growth_rates = historical_data[idx]

        # We are starting with 10,000 users
        users = n_users
        revenue_per_user = r_per_usr
        starting_revenue = users * revenue_per_user

        # Calculate revenue for each period then sum them up
        # to get total revenue across the selected 12 months
        # starting revenue is the first entry used to calculate
        # subsequent months but we drop it from the final calculation

        revenue_per_month = [starting_revenue]

        for growth_rate in selected_growth_rates:
            revenue_per_month.append(
                (1 + growth_rate) * revenue_per_month[-1])

        # Drop the pre-simulation starting month revenue
        total_revenue = np.sum(revenue_per_month[1:])
        results.append(total_revenue)

    return results


def plot_data(results):
    fig,ax=plt.subplots()
    ax = sns.distplot(results, kde=False, norm_hist=True)
    ax.set_xlabel("Annual revenue across simulated 12 months in millions")
    ax.set_ylabel("Probability")
    return fig,ax

def app():
    
    st.title("Subscription based Business model")
    st.subheader("This app helps identify the risk of cash required to keep subscription model running")

    years=st.slider('YEARS',min_value=2.0,max_value=5.0,value=5.0,step=1.0)
    users=st.slider('USERS',min_value=100.0,max_value=10000.0,value=10000.0,step=10.0)
    price= st.slider("Price per user in $",min_value=5,max_value=10,step=5,value=10)
    rounds=st.slider("Number of rounds for simulation",min_value=1000.0,max_value=10000.0,value=10000.0,step=100.0)                     
    fixed_cost=st.text_input("Fixed cost or expense of business",int(1000000))


    N_ROUNDS=rounds
    N_USERS=users
    R_PER_USER=price
    N_YEARS=years

    subscription_data={}
    gen_btn_clck=st.button("Generate Data & Run Simulation")

    if gen_btn_clck:
        
        st.text("N_Users: "+str(N_USERS))
        st.text("Revenue Per user: "+str(R_PER_USER))
        st.text("Number of years: "+str(N_YEARS))

        with st.spinner("generating data...."):

            subscription_data,tol_mon=generate_data(N_YEARS)
            data=pd.DataFrame.from_dict(subscription_data)
            data["Net Growth (%)"] = pd.Series(["{0:.2f}%".format(val * 100) for val in data["Net Growth (%)"]], index = data.index)
            st.dataframe(data)            

        st.success("Done !")

        st.text("Running simulation on subscription data and plotting distribution of same")
        st.text("Simulation rounds: "+str(N_ROUNDS))
        with st.spinner(" running ..."):
            results=run_simulation(subscription_data,int(N_ROUNDS),int(N_YEARS),int(N_USERS),int(R_PER_USER))
        st.success("Done!")                

        fig,ax = plot_data(results)
        ax.set_yticklabels(['{:.9f}'.format(x) for x in ax.get_yticks().tolist()])    
        st.pyplot(fig=fig)    

        avg_val=numerize(np.average(results))

        info_print="**The average or expected annual revenue is $ %s across all %7d simulated scenarios**" %(avg_val,int(N_ROUNDS))
        st.markdown(info_print)

        results.sort()    
        lowest_rev=results[0]
        secondlowest_rev=results[int(0.02*N_ROUNDS)]

        desc=f'''
        The chart above shows the probability distribution for total accumulated annual revenue.
        For example, if we look at the area between USD {str(numerize(lowest_rev))} and USD {str(numerize(secondlowest_rev))},
        we can see the small bars reflecting that such low revenue figures have low likelihood
        of occurring, but are still possible.'''    
        st.text(desc)

        st.subheader("Question: How much cash must be kept aside to cope up the fixed cost or expenses required for running the business?")
        one_percent=int(0.01*N_ROUNDS)
        desc_new=f'''
        To solve our problem, we need to know what annual revenue are we able to achieve or exceed 
        with 99% confidence. We can sort the {str(N_ROUNDS)} simulated total revenue results from 
        lowest to highest and take the {str(one_percent+1)} lowest result.

        {str(one_percent)} out of {str(N_ROUNDS)} (1%) simulated total revenue results are lower 
        than this number, 99% are equal or higher,so we are 99% confident we would achieve or 
        exceed that number. Below is the code to get this number from simulation result 
        '''

        st.text(desc_new)    
        conf_op=results[100]    
        sub_conf=int(fixed_cost)-int(conf_op)        
        code=f'''
        sorted(simulation_results)[100]
        {str(int(conf_op))}
        '''
        st.code(code)        

        desc_conclusion=f'''
        We are 99% confident that we can achieve or exceed $ {str(numerize(conf_op))} over the next 12 months 
        based on this simulation. This would keep us $ {str(numerize(sub_conf))} short of the $ {str(numerize(int(fixed_cost)))} needed 
        to cover our annual expenses.
        
        i.e. {str(numerize(int(fixed_cost)))} - {str(numerize(conf_op))} = {str(numerize(sub_conf))}

        We need to set aside $ {str(numerize(sub_conf))} to be 99% confident of being able to pay the $ {str(numerize(int(fixed_cost)))} million 
        of annual expenses in the next year.
        '''

        st.text(desc_conclusion)



