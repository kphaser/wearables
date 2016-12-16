# -*- coding: utf-8 -*-
import os
import cPickle as pickle
import pandas as pd  # panda's nickname is pd
import numpy as np  # numpy as np
import scipy
import matplotlib.pyplot as plt
from pandas import DataFrame, Series, Categorical
from sklearn import datasets, linear_model
import re


def read_csv_file(filename):
    return(pd.read_csv(filename))
    

## main script
prototype_df = read_csv_file('prototypesurvey_mockdata_112015_v2.csv')

#- One of the key drivers of dissatisfaction with the prototype are issues 
# with the setup and discomfort with the shoe (visible in the unstructured data field “satisfaction_oe”)

str_style_issues = [r"[Dd]idn't.*[sS]tyl",r"[Nn]eed.*style",r"[Ww]ant.*stylish",r"[wW]ish.*style",r"[nN]ot.*stylish"]
str_comfort_issues = [r"[dD]idn't.*support",r'[wW]ish.*light',r'[wW]ish.*comfort',r'[nN]ot.*[Cc]omfort',r'[oO]uch',r'[uU]ncomfortable',r'[bB]ulky',r"[wW]asn't.*comfortable",r"[sS]hoe hurt"]
str_cost_issues = [r'[eE]xpens',r'[mM]oney',r'[cC]ost too']
str_setup_issues = [r'[wWish].*[s]etup',r'[pP]ortal',r'[sS]etup.*difficult',r'[sS]etup.*hard',r'[cC]ouldnt.*[s]etup']
str_feature_issues = [r'[hH]ard.*portal',r"[cC]ouldn't.*portal",r'[nN]ot Interested',r'[mM]eh',r'[oO]kay']

#turn satisfaction_oe into series
satisfaction_oe_series=prototype_df.satisfaction_oe

#count each issue and append to the issues data frame
issues_columns = ['Issue','Count']
issues_df = pd.DataFrame(columns = issues_columns)

def count_issues(str_issue_word_list, str_issue_title):
    issue_count = 0
    for item in str_issue_word_list:
#        print str_issue_title + ": " + item + "count: " + str(satisfaction_oe_series.str.findall(item).str.len().sum())
        issue_count += satisfaction_oe_series.str.findall(item).str.len().sum()
    return pd.DataFrame([[str_issue_title,issue_count]],columns=issues_columns)

issues_df = issues_df.append(count_issues(str_comfort_issues,'Comfort'))
issues_df = issues_df.append(count_issues(str_style_issues,'Style'))
issues_df = issues_df.append(count_issues(str_setup_issues,'Setup'))
issues_df = issues_df.append(count_issues(str_cost_issues,'Cost'))
issues_df = issues_df.append(count_issues(str_feature_issues,'Features'))
issues_df = issues_df.sort(columns = 'Count',ascending=False)

#bar chart to show biggest issues in lowest satisfactions
issues_df.plot(kind='bar',rot=17,legend=False)
plt.xticks(np.arange(5),(issues_df['Issue'].tolist()))
plt.title("Top 5 dissatisfaction issues",fontweight='bold')
plt.ylabel("Dissatisfied Count",fontweight='bold')
plt.show()

#-	Those who have greater levels of satisfaction with the product generally have greater levels of satisfaction with setup 
# Run regression on overall satisfaction and setup
setup_satisfaction = np.array(prototype_df.setup).reshape(1000,1)
satisfaction = np.array(prototype_df.satisfaction).reshape(1000,1)
regr.fit(setup_satisfaction,satisfaction )
cor_coef = np.corrcoef(prototype_df.setup, prototype_df.satisfaction)[0, 1]
r_squared = regr.score(setup_satisfaction,satisfaction )
#Plot the regression
plt.figure()
plt.scatter(setup_satisfaction,satisfaction,   color='black')
plt.plot(satisfaction, regr.predict(satisfaction), color='blue',linewidth=1)
plt.title("Regression of satisfaction with setup on overall satisfaction ")
plt.xlabel("Satisfaction with setup")
plt.ylabel("Overall satisfaction")
plt.ylim(0,10)
plt.xlim(0,10)
str_annotation_r = "r: " + str(round(cor_coef,2))
str_annotation_r_squared = "r^2: " + str(round(r_squared,2))
plt.annotate(str_annotation_r, xy=(2, 8), xytext=(3, 8.5))
plt.annotate(str_annotation_r_squared, xy=(2, 7.5), xytext=(3, 7.75))
plt.show()

# look at relationships between types of user and average satisfaction across satisfied and unsatisfied customers
generally_satisfied_df = prototype_df[prototype_df.satisfaction>=7]
generally_un_satisfied_df = prototype_df[prototype_df.satisfaction<7]

means_columns = ['User type','Satisfied customer','Unsatisfied customer','p-value']
satisfaction_means_df = pd.DataFrame(columns = means_columns)

# t-test of selfrun mean for higher satisfaction vs. lower satisfaction
result = scipy.stats.ttest_ind(prototype_df[prototype_df.satisfaction>=7].selfrun, prototype_df[prototype_df.satisfaction<7].selfrun, equal_var=False)
satisfaction_means_df = satisfaction_means_df.append(pd.DataFrame([["Runner",generally_satisfied_df.selfrun.mean(),generally_un_satisfied_df.selfrun.mean(),result.pvalue]],columns=means_columns))

# t-test of runtrack mean for higher satisfaction vs. lower satisfaction
result = scipy.stats.ttest_ind(prototype_df[prototype_df.satisfaction>=7].runtrack, prototype_df[prototype_df.satisfaction<7].runtrack, equal_var=False)
satisfaction_means_df = satisfaction_means_df.append(pd.DataFrame([["Track runs",generally_satisfied_df.runtrack.mean(),generally_un_satisfied_df.runtrack.mean(),result.pvalue]],columns=means_columns))

# t-test of selfcycle mean for higher satisfaction vs. lower satisfaction
result = scipy.stats.ttest_ind(prototype_df[prototype_df.satisfaction>=7].selfcycle, prototype_df[prototype_df.satisfaction<7].selfcycle, equal_var=False)
satisfaction_means_df = satisfaction_means_df.append(pd.DataFrame([["Cycler",generally_satisfied_df.selfcycle.mean(),generally_un_satisfied_df.selfcycle.mean(),result.pvalue]],columns=means_columns))

#t-test of cycletrack mean for higher satisfaction vs. lower satisfaction
result = scipy.stats.ttest_ind(prototype_df[prototype_df.satisfaction>=7].cycletrack, prototype_df[prototype_df.satisfaction<7].cycletrack, equal_var=False)
satisfaction_means_df = satisfaction_means_df.append(pd.DataFrame([["Track cycling",generally_satisfied_df.cycletrack.mean(),generally_un_satisfied_df.cycletrack.mean(),result.pvalue]],columns=means_columns))

#t-test of community mean for higher satisfaction vs. lower satisfaction
result = scipy.stats.ttest_ind(prototype_df[prototype_df.satisfaction>=7].community, prototype_df[prototype_df.satisfaction<7].community, equal_var=False)
satisfaction_means_df = satisfaction_means_df.append(pd.DataFrame([["Community use",generally_satisfied_df.community.mean(),generally_un_satisfied_df.community.mean(),result.pvalue]],columns=means_columns))

#t-test of portal mean for higher satisfaction vs. lower satisfaction
result = scipy.stats.ttest_ind(prototype_df[prototype_df.satisfaction>=7].portal, prototype_df[prototype_df.satisfaction<7].portal, equal_var=False)
satisfaction_means_df = satisfaction_means_df.append(pd.DataFrame([["Portal use",generally_satisfied_df.portal.mean(),generally_un_satisfied_df.portal.mean(),result.pvalue]],columns=means_columns))

#t-test of ease of satisfaction for higher satisfaction vs. lower satisfaction
result = scipy.stats.ttest_ind(prototype_df[prototype_df.satisfaction>=7].setup, prototype_df[prototype_df.satisfaction<7].setup, equal_var=False)
satisfaction_means_df = satisfaction_means_df.append(pd.DataFrame([["Ease of setup",generally_satisfied_df.setup.mean(),generally_un_satisfied_df.setup.mean(),result.pvalue]],columns=means_columns))
satisfaction_means_df


#-	Style is more important to female respondents than other respondents
style_df            = prototype_df[prototype_df.oneupdate=='Style']
genders_df          = pd.DataFrame({'gender_count' : prototype_df.groupby( [ 'gender'] , as_index=False ).size()})
features_counted_df = pd.DataFrame({'style_count' : style_df.groupby( [ 'gender'] , as_index=False ).size()})
percentages_df      = pd.DataFrame({'Percentages' : (features_counted_df.style_count/genders_df.gender_count)*100}).sort('Percentages')
percentages_df.plot(kind='bar',stacked=False,rot=360,legend=False)
plt.title("Style as most desired feature by gender",fontweight="bold")
plt.xlabel('Gender',fontweight='bold')
plt.ylabel('Percentage',fontweight='bold')
plt.show()

#-	Comfort is particularly important to older respondents
comfort_df          = prototype_df[prototype_df.oneupdate=='Comfort']
ages_df             = pd.DataFrame({'agerange_count' : prototype_df.groupby( [ 'agerange'] , as_index=False ).size()})
features_counted_df = pd.DataFrame({'comfort_count' : comfort_df.groupby( [ 'agerange'] ).size()})
percentages_df      = pd.DataFrame({'Percentages' : (features_counted_df.comfort_count/ages_df.agerange_count)*100}).sort('Percentages')
percentages_df.plot(kind='bar',rot=360,legend=False)
plt.title("Comfort as most desired feature by age range",fontweight='bold')
plt.xlabel('Age Range',fontweight='bold')
plt.ylabel('Percentage',fontweight='bold')
plt.show()


#-	Performance is particularly important to respondents who run frequently
performance_df       = pd.DataFrame(prototype_df[prototype_df.oneupdate=='Performance'])
runners_df           = pd.DataFrame({'runners_count' : prototype_df.groupby  ( [ 'selfrun'] ).size()})
features_counted_df  = pd.DataFrame({'performance_count' : performance_df.groupby( [ 'selfrun'] ).size()})
percentages_df       = pd.DataFrame({'Percentages' : (features_counted_df.performance_count/runners_df.runners_count)*100})
percentages_df.plot(kind='bar',rot=360,legend=False)
plt.title("Performance as most desired feature by runner frequency",fontweight='bold')
plt.xlabel('Runner Frequency',fontweight='bold')
plt.ylabel('Percentage',fontweight='bold')
plt.show()



#demographics analysis

customers_df = read_csv_file('customerdata_mockdata_112515_v2.csv')

#bar chart of top 5 states colored by gender
states_counted = pd.DataFrame({'count' : customers_df.groupby( [ 'state'] , as_index=False ).size()}).reset_index()
top_5_states = pd.DataFrame(states_counted.sort(ascending=False,columns='count')[:5])
top_5_states_df = customers_df[customers_df.state.isin(top_5_states['state'])]
top_5_states_and_gender_df = pd.DataFrame({'count' : top_5_states_df.groupby( [ 'state','gender'] , as_index=False ).size()}).reset_index()
top_5_states_and_gender_df_pivot = top_5_states_and_gender_df.pivot(index='state', columns='gender', values='count')
top_5_states_and_gender_df_pivot.ix[list(top_5_states.state)].plot(kind='bar',stacked=True,color=['red','blue'],rot=17)
plt.title("Top 5 states colored by gender",fontweight='bold')
plt.xlabel("")
plt.show()

#demographics by age range
ages_counted_df        = pd.DataFrame({'count' : customers_df.groupby( [ 'agerange'] , as_index=False ).size()}).reset_index()
ages_counted_df        = ages_counted_df.sort(columns='count')
ages_counted_df.plot(kind='bar',rot=360,legend=False)
plt.title("Count of users by age range",fontweight='bold')
plt.xticks(np.arange(3),ages_counted_df.agerange.tolist())
plt.xlabel('Age Range',fontweight='bold')
plt.ylabel('Count',fontweight='bold')
plt.show()

#-	Those who have larger incomes have generally spent more money on the product
# transform the annual income range to a number using the midpoint - question to consider - what to do with the above 125k?
midpoint_salary = []
for salary in customers_df['annual_income']:
    if salary.find("or more") != -1:
        #do something with the upper bounds - for now putting in 200000
        midpoint_salary.append(150000)
    else:
        salary_range = salary.split('-') 
        upper_bound = re.sub('[\$,]', '', salary_range[1])
        lower_bound = re.sub('[\$,]', '', salary_range[0])
        midpoint_salary.append((float(upper_bound)+ float(lower_bound))/2)
midpoint_salary=np.array(midpoint_salary).reshape(1518,1)

amount_spent = []
for amount in customers_df['prodspent']:
    amount_spent.append(float(re.sub('[\$,]', '', amount)))
amount_spent = np.array(amount_spent).reshape(1518,1)

# Create linear regression object
regr = linear_model.LinearRegression()
regr.fit(midpoint_salary, amount_spent)
r_squared = regr.score(midpoint_salary, amount_spent)
# Plot regression - shows some relationship but also outliers
#question to consider is what to do with the outliers
plt.figure()
plt.scatter(midpoint_salary, amount_spent,  color='black')
plt.plot(midpoint_salary, regr.predict(midpoint_salary), color='blue',linewidth=1)
plt.title("Regression of salary on amount spent",fontweight='bold')
plt.xlabel("Annual income",fontweight='bold')
plt.ylabel("Amount spent",fontweight='bold')
#plt.ylim(100,2000)
#plt.xlim(0,220000)
plt.show()

#would existing customers be interested in a shoe product?
generally_satisfied_df = customers_df[customers_df.prod_satis>=7]
generally_un_satisfied_df = customers_df[customers_df.prod_satis<7]

satisifed_interested_in_shoe = generally_satisfied_df[generally_satisfied_df.shoeinterest>=7]
un_satisifed_interested_in_shoe = generally_un_satisfied_df[generally_un_satisfied_df.shoeinterest>=7]
overall_interested_in_shoe = customers_df[customers_df.shoeinterest>=7]

overall_satisfied_customers_pct = (float(len(generally_satisfied_df.customer_id))/float(len(customers_df.customer_id)))*100
overall_interest_in_shoe_pct = (float(len(overall_interested_in_shoe.customer_id))/float(len(customers_df.customer_id)))*100
satisifed_interested_in_shoe_pct = (float(len(satisifed_interested_in_shoe.customer_id))/float(len(generally_satisfied_df.customer_id)))*100
un_satisifed_interested_in_shoe_pct = (float(len(un_satisifed_interested_in_shoe.customer_id))/float(len(generally_un_satisfied_df.customer_id)))*100

print "Percentage of current satisfied customers: "+ str(round(overall_satisfied_customers_pct,2))+"%"
print "Overall interest in shoe product: " + str(round(overall_interest_in_shoe_pct,2)) + "%"
print "Satisfied customers interest in shoe product: " + str(round(satisifed_interested_in_shoe_pct,2)) + "%"
print "Unsatisfied customers interest in shoe product: " + str(round(un_satisifed_interested_in_shoe_pct,2)) + "%"

