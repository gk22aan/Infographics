# -*- coding: utf-8 -*-
"""
Created on Fri May  5 23:04:15 2023

@author: admin
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

gender_data = pd.read_csv("API_17_DS2_en_csv_v2_5361642.csv",skiprows=4)
country_data = gender_data[gender_data['Country Name'].isin(['China','United States', 'India', 'Japan', 'Germany', 'United Kingdom', 'France', 'Brazil','Russia'])]

indicator_men = country_data[country_data['Indicator Code'].isin(['SL.IND.EMPL.MA.ZS'])]
indicator_data_men = indicator_men.loc[:, ['Country Name','Indicator Code','2019']]

indicator_women = country_data[country_data['Indicator Code'].isin(['SL.IND.EMPL.FE.ZS'])]
indicator_data_women = indicator_women.loc[:, ['Country Name','Indicator Code','2019']]

indicator_data_combined_empl = pd.merge(indicator_data_men, indicator_data_women, left_on='Country Name', right_on='Country Name')
indicator_data_combined_empl.rename(columns={'2019_x': 'Men', '2019_y': 'Women'}, inplace=True)

del indicator_data_combined_empl['Indicator Code_x']
del indicator_data_combined_empl['Indicator Code_y']

indicator_data_combined_c = indicator_data_combined_empl['Country Name']
indicator_data_combined_m = indicator_data_combined_empl['Men']
indicator_data_combined_w = indicator_data_combined_empl['Women']


indicator_men2 = country_data[country_data['Indicator Code'].isin(['SL.TLF.CACT.MA.ZS'])]
indicator_data_men2 = indicator_men2.loc[:, ['Country Name','Indicator Code','2019']]

indicator_women2 = country_data[country_data['Indicator Code'].isin(['SL.TLF.CACT.FE.ZS'])]
indicator_data_women2 = indicator_women2.loc[:, ['Country Name','Indicator Code','2019']]
####################################################################################################################
indicator_women3 = country_data[country_data['Indicator Code'].isin(['SE.SEC.ENRR.FE'])]
indicator_data_women3 = indicator_women3.loc[:, ['Country Name','Indicator Code','2013','2014','2015','2016','2017','2018']]

indicator_data_women3T = indicator_data_women3.T
indicator_data_women3T.rename(columns={ indicator_data_women3T.columns[0]: "Brazil" ,}, inplace = True)
indicator_data_women3T.rename(columns={ indicator_data_women3T.columns[1]: "China" ,}, inplace = True)
indicator_data_women3T.rename(columns={ indicator_data_women3T.columns[2]: "Germany" ,}, inplace = True)
indicator_data_women3T.rename(columns={ indicator_data_women3T.columns[3]: "France" ,}, inplace = True)
indicator_data_women3T.rename(columns={ indicator_data_women3T.columns[4]: "UK" ,}, inplace = True)
indicator_data_women3T.rename(columns={ indicator_data_women3T.columns[5]: "India" ,}, inplace = True)
indicator_data_women3T.rename(columns={ indicator_data_women3T.columns[6]: "Japan" ,}, inplace = True)
indicator_data_women3T.rename(columns={ indicator_data_women3T.columns[7]: "US" ,}, inplace = True)
indicator_data_women3T = indicator_data_women3T.reset_index()
indicator_data_women3T = indicator_data_women3T.drop(labels=[0,1], axis=0)
#######################################################################################################

indicator_men3 = country_data[country_data['Indicator Code'].isin(['SE.SEC.ENRR.MA'])]
indicator_data_men3 = indicator_men3.loc[:, ['Country Name','Indicator Code','2013','2014','2015','2016','2017','2018']]

indicator_data_men3T = indicator_data_men3.T
indicator_data_men3T.rename(columns={ indicator_data_men3T.columns[0]: "Brazil" ,}, inplace = True)
indicator_data_men3T.rename(columns={ indicator_data_men3T.columns[1]: "China" ,}, inplace = True)
indicator_data_men3T.rename(columns={ indicator_data_men3T.columns[2]: "Germany" ,}, inplace = True)
indicator_data_men3T.rename(columns={ indicator_data_men3T.columns[3]: "France" ,}, inplace = True)
indicator_data_men3T.rename(columns={ indicator_data_men3T.columns[4]: "UK" ,}, inplace = True)
indicator_data_men3T.rename(columns={ indicator_data_men3T.columns[5]: "India" ,}, inplace = True)
indicator_data_men3T.rename(columns={ indicator_data_men3T.columns[6]: "Japan" ,}, inplace = True)
indicator_data_men3T.rename(columns={ indicator_data_men3T.columns[7]: "US" ,}, inplace = True)

indicator_data_men3T = indicator_data_men3T.reset_index()
indicator_data_men3T = indicator_data_men3T.drop(labels=[0,1], axis=0)
########################################################################################################################
indicator_men4 = country_data[country_data['Indicator Code'].isin(['SL.EMP.WORK.MA.ZS'])]
indicator_data_men4 = indicator_men4.loc[:, ['Country Name','Indicator Code','2019']]

indicator_women4 = country_data[country_data['Indicator Code'].isin(['SL.EMP.WORK.FE.ZS'])]
indicator_data_women4 = indicator_women4.loc[:, ['Country Name','Indicator Code','2019']]

indicator_data_combined_gender= pd.concat([indicator_data_men4, indicator_data_women4])
indicator_data_combined_gender["Indicator Code"] = indicator_data_combined_gender["Indicator Code"].apply(lambda x: x.replace("SL.EMP.WORK.MA.ZS", "Men"))
indicator_data_combined_gender["Indicator Code"] = indicator_data_combined_gender["Indicator Code"].apply(lambda x: x.replace("SL.EMP.WORK.FE.ZS", "Women"))
indicator_data_combined_gender = indicator_data_combined_gender.reset_index()
indicator_data_combined_gender['2019'] = indicator_data_combined_gender['2019'].astype('float')
indicator_data_combined_gender.loc[indicator_data_combined_gender['Indicator Code'] == 'Women', ['2019'] ] = indicator_data_combined_gender['2019'] * -1
########################################################################################################################
pkmn_type_colors = ['#78C850',  # Grass
                    '#F08030',  # Fire
                    '#6890F0',  # Water
                    '#A8B820',  # Bug
                    '#A8A878',  # Normal
                    '#A040A0',  # Poison
                    '#F8D030',  # Electric
                    '#E0C068',  # Ground
                    '#EE99AC',  # Fairy
                    '#C03028',  # Fighting
                    '#F85888',  # Psychic
                    '#B8A038',  # Rock
                    '#705898',  # Ghost
                    '#98D8D8',  # Ice
                    '#7038F8',  # Dragon
                   ]

###############################################################################
# Display the plot
fig = plt.figure(figsize=(20,20),dpi = 300 ,linewidth=10, edgecolor="black")
plt.style.use('classic')
#plt.style.use('dark_background')
fig.suptitle('Gender Equality - Are Women Really equal in all fronts?' ,fontstyle = 'italic'
             ,color = 'black',fontsize=40,fontname='Georgia')
#plt.rcParams['font.sans-serif'] = 'italic'
gs = fig.add_gridspec(nrows =4 , ncols = 6)
################################################################################################################
ax1 = fig.add_subplot(gs[0, 1:5])
group_col = 'Indicator Code'
order_of_bars = ['India','China','Brazil','United Kingdom', 'France',
      'Japan','Germany','United States']
#order_of_bars = ['']
#order_of_bars = indicator_data_combined_gender['Country Name'].unique()[::-1]
#colors = [plt.cm.Spectral(i/float(len(indicator_data_combined_gender[group_col].unique())-1)) for i in range(len(indicator_data_combined_gender[group_col].unique()))]
colors = ['dodgerblue','deeppink']
for c, group in zip(colors, indicator_data_combined_gender[group_col].unique()):
    sns.barplot(x='2019', y='Country Name', data=indicator_data_combined_gender.loc[indicator_data_combined_gender[group_col]==group, :], order=order_of_bars, color=c, label=group)

# Decorations
#plt.xlabel("$Users$")
#plt.ylabel("Stage of Purchase")
plt.yticks(fontsize=12)
#plt.title("Population Pyramid of the Marketing Funnel", fontsize=22)
plt.legend()
################################################################################################################
ax2 = fig.add_subplot(gs[1, 1:3])
#sns.countplot(x= indicator_nm['Country Name'], data = indicator_data, palette=pkmn_type_colors)
def func(pct):
  return "{:1.1f}%".format(pct)
colors  = ['deeppink']
explode = [0.1,0.2,0.1,0.1,0.2,0.1,0.1,0.1]
#plt.pie(indicator_data_women2['2019'], labels=indicator_data_combined_c, autopct=lambda pct: func(pct), explode=explode, shadow=True, startangle=45, colors = colors)
import matplotlib.patches as patches
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
img = plt.imread('female_pie.png')
imagebox = OffsetImage(img, zoom=0.2)
imagebox.image.axes = ax2
patches, texts, autotexts = ax2.pie(indicator_data_women2['2019'], labels=indicator_data_combined_c, autopct=lambda pct: func(pct), explode=explode, shadow=True, startangle=45, colors = colors)# autopct='%1.1f%%',
xy = (autotexts[1].get_position()[0], autotexts[1].get_position()[1])
ab = AnnotationBbox(imagebox, xy,
                    xybox=(50., -50.),
                    xycoords='data',
                    boxcoords="offset points",
                    pad=0.5,
                    bboxprops={'edgecolor':'white'}
                    )
ax2.add_artist(ab)
ax2.axis('equal')
for t in autotexts:
    t.remove()


################################################################################################################
ax3 = fig.add_subplot(gs[1, 3:5])
colors = ['dodgerblue']
#plt.pie(indicator_data_men2['2019'], labels=indicator_data_combined_c, autopct=lambda pct: func(pct), explode= explode , shadow=True, startangle=45, colors = colors)
import matplotlib.patches as patches
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)

img = plt.imread('male_pie.png')
imagebox = OffsetImage(img, zoom=0.2)
imagebox.image.axes = ax3
patches, texts, autotexts = ax3.pie(indicator_data_men2['2019'], labels=indicator_data_combined_c, autopct=lambda pct: func(pct), explode=explode, shadow=True, startangle=45, colors = colors)# autopct='%1.1f%%',
xy = (autotexts[1].get_position()[0], autotexts[1].get_position()[1])
ab = AnnotationBbox(imagebox, xy,
                    xybox=(50., -50.),
                    xycoords='data',
                    boxcoords="offset points",
                    pad=0.5,
                    bboxprops={'edgecolor':'white'}
                    )
ax3.add_artist(ab)
ax3.axis('equal')
for t in autotexts:
    t.remove()
#################################################################################################################
ax4 = fig.add_subplot(gs[2, 1:3])
columns = indicator_data_women3T.columns[1:]
labs = columns.values.tolist()
mycolors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:brown', 'tab:grey', 'tab:pink', 'tab:olive']
# Prepare data
x  = indicator_data_women3T['index'].values.tolist()
y0 = indicator_data_women3T[columns[0]].values.tolist()
y1 = indicator_data_women3T[columns[1]].values.tolist()
y2 = indicator_data_women3T[columns[2]].values.tolist()
y3 = indicator_data_women3T[columns[3]].values.tolist()
y4 = indicator_data_women3T[columns[4]].values.tolist()
y5 = indicator_data_women3T[columns[5]].values.tolist()
y6 = indicator_data_women3T[columns[6]].values.tolist()
y7 = indicator_data_women3T[columns[7]].values.tolist()
y = np.vstack([y0, y2, y4, y6, y7, y5, y1, y3])

# Plot for each column
labs = columns.values.tolist()
ax = plt.gca()
ax.stackplot(x, y, labels=labs, colors=mycolors, alpha=0.8)


#ax.set_title('School enrollment, secondary, female', fontsize=18)
ax.set(ylim=[0, 1500])
ax.legend(fontsize=10, ncol=2)
#plt.xticks(x[::1], fontsize=10, horizontalalignment='center')
#plt.yticks(np.arange(100, 1100, 200), fontsize=10)
#plt.xlim(x[0], x[-1])

# Lighten borders
plt.gca().spines["top"].set_alpha(0)
plt.gca().spines["bottom"].set_alpha(.3)
plt.gca().spines["right"].set_alpha(0)
plt.gca().spines["left"].set_alpha(.3)
#####################################################################################################
ax5 = fig.add_subplot(gs[2, 3:5])
columns = indicator_data_men3T.columns[1:]
labs = columns.values.tolist()
mycolors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:brown', 'tab:grey', 'tab:pink', 'tab:olive']
# Prepare data
x  = indicator_data_men3T['index'].values.tolist()
y0 = indicator_data_men3T[columns[0]].values.tolist()
y1 = indicator_data_men3T[columns[1]].values.tolist()
y2 = indicator_data_men3T[columns[2]].values.tolist()
y3 = indicator_data_men3T[columns[3]].values.tolist()
y4 = indicator_data_men3T[columns[4]].values.tolist()
y5 = indicator_data_men3T[columns[5]].values.tolist()
y6 = indicator_data_men3T[columns[6]].values.tolist()
y7 = indicator_data_men3T[columns[7]].values.tolist()
y = np.vstack([y0, y2, y4, y6, y7, y5, y1, y3])

# Plot for each column
labs = columns.values.tolist()
ax = plt.gca()
ax.stackplot(x, y, labels=labs, colors=mycolors, alpha=0.8)

#ax.set_title('School enrollment, secondary, female', fontsize=18)
ax.set(ylim=[0, 1500])
ax.legend(fontsize=10, ncol=2)
#plt.xticks(x[::1], fontsize=10, horizontalalignment='center')
#plt.yticks(np.arange(100, 1100, 200), fontsize=10)
#plt.xlim(x[0], x[-1])

# Lighten borders
plt.gca().spines["top"].set_alpha(0)
plt.gca().spines["bottom"].set_alpha(.3)
plt.gca().spines["right"].set_alpha(0)
plt.gca().spines["left"].set_alpha(.3)

######################################################################################################
ax6 = fig.add_subplot(gs[3, 1:5])
ax6.barh(indicator_data_combined_c,indicator_data_combined_m, color = 'dodgerblue',align='center', height=.5,label='Men')
ax6.barh(indicator_data_combined_c,indicator_data_combined_w, color = 'deeppink',align='center', height=.5,label='Women')
ax6.set_yticks(indicator_data_combined_c)
ax6.legend()
#######################################################################################################
ax3.set_yticks([])
#ax4.set_yticks([])
ax5.set_yticks([])

ax1.set_title('Wage and salaried workers, male & Female', fontsize=22)
ax2.set_title('Labor force participation rate, Female',fontsize=22)
ax3.set_title('Labor force participation rate, Male',fontsize=22)
ax4.set_title('Secondary School Enrollment-Female', fontsize=22)
ax5.set_title('Secondary School Enrollment-Male', fontsize=22)
ax6.set_title('Employment in industry,(% of Male and female employment)', fontsize=22)
ax1.set_facecolor('yellow')
ax2.set_facecolor('yellow')
ax3.set_facecolor('yellow')
ax4.set_facecolor('yellow')
ax5.set_facecolor('yellow')
ax6.set_facecolor('yellow')

###########################Figure######################################################################
im = plt.imread('female11.png') # insert local path of the image.
newax = fig.add_axes([0.0,0.0,0.1,0.1], anchor='NE', zorder=-3)
newax.imshow(im)
newax.axis('off')
im = plt.imread('female11.png') # insert local path of the image.
newax = fig.add_axes([0.0,0.1,0.1,0.1], anchor='NE', zorder=-3)
newax.imshow(im)
newax.axis('off')
im = plt.imread('female11.png') # insert local path of the image.
newax = fig.add_axes([0.0,0.18,0.1,0.1], anchor='NE', zorder=-3)
newax.imshow(im)
newax.axis('off')
im = plt.imread('female11.png') # insert local path of the image.
newax = fig.add_axes([0.0,0.26,0.1,0.1], anchor='NE', zorder=-3)
newax.imshow(im)
newax.axis('off')
im = plt.imread('female11.png') # insert local path of the image.
newax = fig.add_axes([0.0,0.34,0.1,0.1], anchor='NE', zorder=-3)
newax.imshow(im)
newax.axis('off')
im = plt.imread('female11.png') # insert local path of the image.
newax = fig.add_axes([0.0,0.42,0.1,0.1], anchor='NE', zorder=-3)
newax.imshow(im)
newax.axis('off')
im = plt.imread('female11.png') # insert local path of the image.
newax = fig.add_axes([0.0,0.50,0.1,0.1], anchor='NE', zorder=-3)
newax.imshow(im)
newax.axis('off')
im = plt.imread('female11.png') # insert local path of the image.
newax = fig.add_axes([0.0,0.58,0.1,0.1], anchor='NE', zorder=-3)
newax.imshow(im)
newax.axis('off')
im = plt.imread('female11.png') # insert local path of the image.
newax = fig.add_axes([0.0,0.66,0.1,0.1], anchor='NE', zorder=-3)
newax.imshow(im)
newax.axis('off')
im = plt.imread('female11.png') # insert local path of the image.
newax = fig.add_axes([0.0,0.74,0.1,0.1], anchor='NE', zorder=-3)
newax.imshow(im)
newax.axis('off')
im = plt.imread('female11.png') # insert local path of the image.
newax = fig.add_axes([0.0,.82,0.1,0.1], anchor='NE', zorder=-3)
newax.imshow(im)
newax.axis('off')
im = plt.imread('female11.png') # insert local path of the image.
newax = fig.add_axes([0.0,.90,0.1,0.1], anchor='NE', zorder=-3)
newax.imshow(im)
newax.axis('off')
##############################################################################################################
im = plt.imread('female11.png') # insert local path of the image.
newax = fig.add_axes([0.00,0.98,0.1,0.1], anchor='NE', zorder=-3)
newax.imshow(im)
newax.axis('off')
im = plt.imread('female11.png') # insert local path of the image.
newax = fig.add_axes([0.11,0.98,0.1,0.1], anchor='NE', zorder=-3)
newax.imshow(im)
newax.axis('off')
im = plt.imread('female11.png') # insert local path of the image.
newax = fig.add_axes([0.22,0.98,0.1,0.1], anchor='NE', zorder=-3)
newax.imshow(im)
newax.axis('off')
im = plt.imread('female11.png') # insert local path of the image.
newax = fig.add_axes([0.33,0.98,0.1,0.1], anchor='NE', zorder=-1)
newax.imshow(im)
newax.axis('off')
im = plt.imread('female11.png') # insert local path of the image.
newax = fig.add_axes([0.44,0.98,0.1,0.1], anchor='NE', zorder=-1)
newax.imshow(im)
newax.axis('off')
im = plt.imread('female11.png') # insert local path of the image.
newax = fig.add_axes([0.55,0.98,0.1,0.1], anchor='NE', zorder=-1)
newax.imshow(im)
newax.axis('off')
im = plt.imread('female11.png') # insert local path of the image.
newax = fig.add_axes([0.66,0.98,0.1,0.1], anchor='NE', zorder=-1)
newax.imshow(im)
newax.axis('off')
im = plt.imread('female11.png') # insert local path of the image.
newax = fig.add_axes([0.77,0.98,0.1,0.1], anchor='NE', zorder=-1)
newax.imshow(im)
newax.axis('off')
import warnings
warnings.filterwarnings("ignore")
#################################################################################################################

#################################################################################################################
#ax1.text(0.1, 0.5, 'Begin text', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
fig.tight_layout()
#fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)
# Display the plot
plt.style.use('classic')
#fig.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
#ax1.set_facecolor('pink')
fig.subplots_adjust(top=0.8999999)
plt.show()
#fig.savefig('myplot.png', dpi=300, edgecolor=fig.get_edgecolor())
