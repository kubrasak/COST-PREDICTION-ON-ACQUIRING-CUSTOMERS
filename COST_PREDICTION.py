#########################################################
#################ADD LIBRARIES####################
#########################################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,AdaBoostRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
#from helpers.data_prep import *
#from helpers.eda import *
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler




warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)



################################################
#  Exploratory Data Analysis
################################################
df = pd.read_csv(r"C:\Users\KUBRA SAK\PycharmProjects\pythonProject\dsmlbc_9_gulbuke\Homeworks\Kubra_Sak\Project\media prediction and its cost.csv")
new_df=df.copy()


#food_category --> ürün kategorisi
#food_department -->ürünün satıldığı departman
#food_family     -->ürün ailesi (3 farklı değer alıyor Drink,Food,Non-Consumable)
#store_sales(in millions) -->Ürünü ne kadara satıyorum  (SRP*unit_sales(in millions) )
#store_cost(in millions) --> Ürünün maaliyeti
#unit_sales(in millions)  -->ürünün adedini temsil ediyor
#promotion_name          -->promosyon ismi
#sales_country           -->satış yapılan mağazanın ülkesi
#marital_status          -->müşteri evlilik durumu
#gender                  -->müşteri cinsiyeti
#total_children          -->müşterinin sahip olduğu çocuk sayısı
#education               -->müşterinin eğitim durumu
#member_card             -->müşterinin üyelik bilgisi
#occupation              -->müşterinin mesleği ("Manual","Skilled Manual","Professional","Management","Clerical"
#houseowner              -->müşterinin ev sahibi olup olmadığı ('Y' VE 'N' değerleri)
#avg_cars_at home(approx) -->müşterinin evinde sahip olduğu araç sayısı
#avg. yearly_income       -->müşterinin ortalama yıllık gelirinin aralığı
#num_children_at_home     -->müşterinin evde onlaral birlikte yaşayan çocul sayısı
#avg_cars_at home(approx).1 -->dublike kolon
#brand_name              -->ürünün markası
#SRP                     -->önerilen satış fiyatı
#gross_weight           -->ürünün brüt ağırlığı
#net_weight             -->ürünün net ağırlığı
#recyclable_package     -->ürünün geridönüştürülebilen ambalaj bilgisi
#low_fat                -->ürünün düşük kalorili olup olmaması (1 düşük kalorili demek 0 hayır demek oluyor)
#units_per_case         -->kutudan çıkan adet
#store_type             -->mağazanın tipi("Deluxe Supermarket","Supermarket","Gourmet Supermarket","Mid-Size Grocery","Small Grocery"
#store_city             -->mağazın şehri
#store_state            -->mağazanın ilçesi
#store_sqft             -->mağazanın alanı
#grocery_sqft           -->mağazanın manav bölümünün alanı
#frozen_sqft           -->mağazanın dondurulmuş bölümün alanı (meat_sqft ile 1-1 aynı)
#meat_sqft             -->mağazanın dondurulmuş bölümün alanı (frozen_sqft ile 1-1 aynı)
#coffee_bar            -->mağazada coffe bar bölümü var mı?(1 var 0 yok)
#video_store           -->mağazada video store bölümü var mı?(1 var 0 yok)
#salad_bar             -->mağazada salad bar bölümü var mı?(1 var 0 yok)(prepared_food ile 1-1 aynı değeri alıyor)
#prepared_food         -->mağazada hazırgıda  bölümü var mı?(1 var 0 yok)(salad_food ile 1-1 aynı değeri alıyor)
#florist               -->mağazada çiçekçi bölümü var mı?(1 var 0 yok)
#media_type            -->tanıtımlar nereden yapılıyor, medya kanalları
#cost                  -->medya harcamaları (Hedef değişken)


def check_df(dataframe, head=5):
    """
    Dataframe ile ilgili özet bilgiler sunar.

    Parameters
    ----------
    dataframe: dataframe
      Incelenecek datasettir.

    head: int
      Dataset'den incelenmek istenen gözlem sayısıdır.

    Returns
    -------

    """
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0.10, 0.25, 0.50,0.75,0.90, 0.95, 0.99]).T)

check_df(df)


###################
# grab_col_names
###################

df.drop(columns=['avg_cars_at home(approx).1'],inplace=True)

def grab_col_names(dataframe, cat_th=5, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

###threshold 20 KEN
#Observations: 60428
#Variables: 39
#cat_cols: 24 -->('unit_sales(in millions)'(6), 'total_children'(6), 'avg_cars_at home(approx)'(5), 'num_children_at_home'(6))
#num_cols: 11
#cat_but_car: 4
#num_but_cat: 11


##THERSHOLD 5 IKEN
##Observations: 60428
##Variables: 39
##cat_cols: 20
##num_cols: 15
##cat_but_car: 4 ##['food_category', 'food_department', 'promotion_name', 'brand_name']
##num_but_cat: 7


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in num_cols:
    print(col)
    num_summary(df, col, plot=True)

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "cost", col)

def target_summary_with_cat(dataframe, target, categorical_col):
    print(dataframe.groupby(categorical_col).agg({target : ["mean",np.median,"count"]}))


for col in cat_cols:
    target_summary_with_cat(df, "cost", col)

def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)


# Correlation of numerical variables with each other
correlation_matrix(df, num_cols)

def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (40, 40)})
        sns.heatmap(corr, cmap="RdBu",annot=True)
        plt.show()
    return drop_list

high_correlated_cols(df,plot=True)

#############################################
# Catch Outliers
#############################################


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False



for col in num_cols:
    print(col, check_outlier(df, col))

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df=df.drop(columns=["salad_bar","frozen_sqft"])
#############################################
# Feature Engineering (Değişken Mühendisliği)
#############################################

#NEW_MARGIN -->üründen elde edilen kar
df["NEW_MARGIN"]=(df["store_sales(in millions)"]-df["store_cost(in millions)"])/df["store_cost(in millions)"]

#NEW_UNIT_PRICE_AT_BULK_SALES -->Unit price in bulk sale (there is more than one product in the box)
df["NEW_UNIT_PRICE_AT_BULK_SALES"]=df["store_sales(in millions)"]/(df["unit_sales(in millions)"]*df["units_per_case"])

#NEW_COST_SALES_RATIO-->ratio of cost to profit
#df["NEW_COST_SALES_RATIO"]=df["store_cost(in millions)"]/df["store_sales(in millions)"]

#NEW_GROSS_NET_RATIO -->Ratio of gross to net
df["NEW_GROSS_NET_RATIO"]=df["net_weight"]/df["gross_weight"]

#NEW_WELFARE_LEVEL  -->Welfare Level
df.loc[df["avg. yearly_income"] == "$10K - $30K", "NEW_WELFARE_LEVEL"]=1
df.loc[df["avg. yearly_income"] == "$30K - $50K", "NEW_WELFARE_LEVEL"]=2
df.loc[df["avg. yearly_income"] == "$50K - $70K", "NEW_WELFARE_LEVEL"]=3
df.loc[df["avg. yearly_income"] == "$70K - $90K", "NEW_WELFARE_LEVEL"]=4
df.loc[df["avg. yearly_income"] == "$90K - $110K", "NEW_WELFARE_LEVEL"]=5
df.loc[df["avg. yearly_income"] == "$110K - $130K", "NEW_WELFARE_LEVEL"]=6
df.loc[df["avg. yearly_income"] == "$130K - $150K", "NEW_WELFARE_LEVEL"]=7
df.loc[df["avg. yearly_income"] == "$150K +", "NEW_WELFARE_LEVEL"]=8

#NEW_MEMBER_CARD_LEVEL -->Member card Level
df.loc[df["member_card"] == "Normal", "NEW_MEMBER_CARD_LEVEL"] = 1
df.loc[df["member_card"] == "Bronze", "NEW_MEMBER_CARD_LEVEL"] = 2
df.loc[df["member_card"] == "Silver", "NEW_MEMBER_CARD_LEVEL"] = 3
df.loc[df["member_card"] == "Golden", "NEW_MEMBER_CARD_LEVEL"] = 4

#NEW_EDUCATION_LEVEL  -->Education Level
df.loc[df["education"] == "Partial High School", "NEW_EDUCATION_LEVEL"] = 1
df.loc[df["education"] == "High School Degree", "NEW_EDUCATION_LEVEL"] = 2
df.loc[df["education"] == "Partial College", "NEW_EDUCATION_LEVEL"] = 3
df.loc[df["education"] == "Bachelors Degree", "NEW_EDUCATION_LEVEL"] = 4
df.loc[df["education"] == "Graduate Degree", "NEW_EDUCATION_LEVEL"] = 5

#NEW_STORE_TYPE_LEVEL -->STORE TYPE LEVEL
df.loc[df["store_type"] == "Small Grocery", "NEW_STORE_TYPE_LEVEL"] = 1
df.loc[df["store_type"] == "Mid-Size Grocery", "NEW_STORE_TYPE_LEVEL"] = 2
df.loc[df["store_type"] == "Supermarket", "NEW_STORE_TYPE_LEVEL"] = 3
df.loc[df["store_type"] == "Gourmet Supermarket", "NEW_STORE_TYPE_LEVEL"] = 4
df.loc[df["store_type"] == "Deluxe Supermarket", "NEW_STORE_TYPE_LEVEL"] = 5

#Media Type --> Splitting the values in the media variable and producing separate columns TV, Daily Paper, Radio etc.
temp=df["media_type"].value_counts()
media_types=temp.index.to_numpy()
new_media_types=[]
for m in media_types:
    x=m.split(",")
    for i in x:
        i.lstrip()
        print(i.lstrip())
        new_media_types.append(i.lstrip())

media_list=list(set(new_media_types))
df[media_list]=np.zeros(9,dtype = int)

for media in media_list:
    print(media)
    df[media]=df["media_type"].apply(lambda x: 1 if media in x else 0 )

#NEW_ONLINE_CHANNEL --> Have you advertised online channels?
#df["NEW_ONLINE_CHANNEL"]=df.apply(lambda x: 1 if x["TV"]==1 or x["Radio"]==1 or x["Bulk Mail"]==1 else 0 ,axis=1)

df["SUM_CHANNEL"]=df['Daily Paper']+df['Sunday Paper']+df['Product Attachment']+df['TV']\
                  +df['Radio']+df['Cash Register Handout']+df['Street Handout']+df['In-Store Coupon']+\
                df['Bulk Mail']

df["RATIO_ONLINE_CHANNEL"]=(df["TV"]+ df["Radio"]+ df["Bulk Mail"])/df["SUM_CHANNEL"]


###FEATURES RELATED TO PROMOTION
#NEW_PROMOTION_DAY -->Words such as "day" and "week" in the promotion(Indicates a time limit)
df["NEW_PROMOTION_DAY"] = df["promotion_name"].apply(lambda x: 1 if any(word in x.upper() for word in ["WEEKEND","DAY"]) else 0)

#NEW_PROMOTION_SAVER -->Words such as "SAVING" and "SAVERS" in the promotion(Indicates saving)
df["NEW_PROMOTION_SAVER"] = df["promotion_name"].apply(lambda x: 1 if any(word in x.upper() for word in ["SAVERS","SAVINGS"]) else 0)

#split store_sqt by sqrt ones
#NEW_GROCERY_RATIO -->Ratio of grocery area to the whole store
#df["NEW_GROCERY_RATIO"]=df["grocery_sqft"]/df["store_sqft"]---> Correlation is high, we did not use

#NEW_MEAT_RATIO -->Ratio of meat area to whole store
#df["NEW_GROCERY_RATIO"]=df["meat_sqft"]/df["store_sqft"]-->Correlation is high, we did not use
#NEW_MEAT_GROCERY_RATIO
df["NEW_MEAT_GROCERY_RATIO"]=df["meat_sqft"]/df["grocery_sqft"]

#SALAD_BAR,COFFEE_BAR  VS additional fields about store
#'video_store', 'salad_bar', 'prepared_food', 'florist',"coffee_bar"
#NEW_EXTRA_AREAS -->How many additional fields are there
df["NEW_EXTRA_AREAS"]=df["video_store"]+df["coffee_bar"]+df["prepared_food"]+df["florist"]

#NEW_STORE_NUM_IN_STATE    --> How many stores are in your state?
#NEW_STORE_REVENUE_IN_STATE  -->Total income in state (temp variable will be deleted later)
#NEW_RATIO_SALES_TO_STATE --->Income rate by state
temp=new_df.copy()
temp["TEMP_CITY_TYPE"]=temp["store_city"]+temp["store_type"]
data=temp.groupby(["store_state"]).agg({"TEMP_CITY_TYPE": ["nunique"],
                                        "store_sales(in millions)":"sum"})
data.columns=["NEW_STORE_NUM_IN_STATE","NEW_STORE_REVENUE_IN_STATE"]
data=data.reset_index(col_level=1)
df=df.merge(data ,on="store_state")
df["NEW_RATIO_SALES_TO_STATE"]=df["store_sales(in millions)"]/df["NEW_STORE_REVENUE_IN_STATE"]



#NEW_IS_CAR -->DO YOU HAVE A CAR?
df["NEW_IS_CAR"]=df["avg_cars_at home(approx)"].apply(lambda x: 1 if x>0 else 0)


#FEATURE INTERACTION
#COST BASED BRAND
#NEW_FOOD_NUM_IN_BRAND_AT_STORE -->How many products of the Brand are there in the relevant store?
#NEW_REVENUE_IN_BRAND_AT_STORE -->Income of the Brand in the Related Store (To be used as a temporary variable)
#NEW_RATIO_REVENUE_TO_BRAND_AT_STORE --> How much is the total revenue of the relevant brand in the store where it is located?
temp=new_df.copy()
temp["TEMP_FOOD"]=temp["food_category"]+temp["food_department"]
data=temp.groupby(["brand_name","store_city","store_type"]).agg({"TEMP_FOOD": ["nunique"],
                                        "store_sales(in millions)":"sum"})
data.columns=["NEW_FOOD_NUM_IN_BRAND_AT_STORE","NEW_REVENUE_IN_BRAND_AT_STORE"]
data=data.reset_index(col_level=1)
df=df.merge(data ,on=["brand_name","store_city","store_type"])
df["NEW_RATIO_REVENUE_TO_BRAND_AT_STORE"]=df["store_sales(in millions)"]/df["NEW_REVENUE_IN_BRAND_AT_STORE"]


#NEW_FOOD_NUM_IN_BRAND -->How many products belong to the brand
#NEW_REVENUE_IN_BRAND -->Brand revenue (to be used as a temporary variable)
#NEW_BRAND_BASED_REVENUE_BY_STORE --> How much is the total revenue of the relevant brand in the store ?
data=temp.groupby(["brand_name"]).agg({"TEMP_FOOD": ["nunique"],
                                        "store_sales(in millions)":"sum"})
data.columns=["NEW_FOOD_NUM_IN_BRAND","NEW_REVENUE_IN_BRAND"]
data=data.reset_index(col_level=1)
df=df.merge(data ,on=["brand_name"])
df["NEW_BRAND_BASED_REVENUE_BY_STORE"]=df["NEW_REVENUE_IN_BRAND_AT_STORE"]/df["NEW_REVENUE_IN_BRAND"]


#NEW_MARRIED_CHILDREN_RELATION -->STATUS OF BEING MARRIED WITH CHILDREN
df.loc[((df["marital_status"]=='M') & (df["total_children"]>0)), "NEW_MARRIED_CHILDREN_RELATION"]= "NEW_MARRIED_PARENT"
df.loc[((df["marital_status"]=='M') & (df["total_children"]==0)), "NEW_MARRIED_CHILDREN_RELATION"]="NEW_MARRIED_CHILDLESS"
df.loc[((df["marital_status"]=='S') & (df["total_children"]>0)), "NEW_MARRIED_CHILDREN_RELATION"]="NEW_SINGLE_PARENT"
df.loc[((df["marital_status"]=='S') & (df["total_children"]==0)), "NEW_MARRIED_CHILDREN_RELATION"]="NEW_SINGLE_CHILDLESS"

#NEW_RATIO_CHILDREN_AT_HOME  -->Ratio of total children to children at home
df["NEW_RATIO_CHILDREN_AT_HOME"]=df["num_children_at_home"]/df["total_children"].replace(0,1)


df.shape
##DROP list
onehot_df=df.copy()
df=df.drop(columns=["store_cost(in millions)","net_weight",\
                    "avg. yearly_income","member_card",\
                    "education","store_type","media_type","promotion_name",\
                    "food_category","food_department","NEW_REVENUE_IN_BRAND",\
                    "NEW_REVENUE_IN_BRAND_AT_STORE","store_state","store_city",\
                    "NEW_STORE_REVENUE_IN_STATE","brand_name","num_children_at_home"\
                    ,"meat_sqft","grocery_sqft"])

#df=onehot_df.copy() Geri dönme ile alakalo
df.head()

#onehot food_family,houseowner,sales_country,occupation,NEW_MARRIED_CHILDREN_RELATION
cat_cols, num_cols, cat_but_car = grab_col_names(df,cat_th=4)
#marital_status,gender,
#NEW_MEMBER_CARD_LEVEL, 'NEW_STORE_NUM_IN_STATE'
drop_columns=["marital_status","NEW_IS_CAR",\
              "gender", 'occupation', 'houseowner','NEW_MARRIED_CHILDREN_RELATION',"food_family",\
              "NEW_FOOD_NUM_IN_BRAND","florist","prepared_food","coffee_bar",\
              "SRP"]

#NEW_FOOD_NUM_IN_BRAND_AT_STORE meat_sqft/grocery_sqft florist,
# prepared_food,video_store,coffee_bar,NEW_ONLINE_CHANNEL,
# sales_country_Mexico,sales_country_USA,SRP
df=df.drop(columns=drop_columns)
ohe_cols=["sales_country"]
df.head()

#ohe_cols=["food_family","sales_country","marital_status","gender", 'occupation', 'houseowner','NEW_MARRIED_CHILDREN_RELATION']
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, ohe_cols, drop_first=True)

df=df.drop(columns=["sales_country_Mexico"])
df=df.drop(columns=["video_store"])



#############################################
# Standart Scaler
#############################################
cat_cols, num_cols, cat_but_car = grab_col_names(df,cat_th=4)
num_cols=[x for x in num_cols if x!='cost']
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

df.head()
df.shape





######################################################
# Base Models
######################################################
y = df["cost"]
X = df.drop(["cost"], axis=1)

#Randomforest Tahminleme

# Verinin eğitim ve tet verisi olarak bölünmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

rf_model = RandomForestRegressor(max_depth=None, max_features='auto',min_samples_split=2,n_estimators=500).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, y_pred)))


df["y_all_pred"] = rf_model.predict(X)
result=pd.merge(new_df,df, left_index=True, right_index=True)

result.to_csv("output.csv")

new_df.groupby("promotion_name",)
################################################
# Feature Importance
################################################

def plot_importance(model, features, num=len(X_train), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X_train)

##KORELASYON INCELEMESI
dataCorr = df.corr(method='pearson')
dataCorr = dataCorr.mask(np.tril(np.ones(dataCorr.shape)).astype(np.bool))

dataCorr = dataCorr[abs(dataCorr) > 0.7].stack().reset_index()
print(dataCorr)


lgbm_model = LGBMRegressor(random_state=46).fit(X_train, y_train)
y_pred = lgbm_model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, y_pred)))
plot_importance(lgbm_model, X)


models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor())]
          # ("CatBoost", CatBoostRegressor(verbose=False))]



for name, regressor in models:
    print(name)
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X_train, y_train, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

df['cost'].mean()
df['cost'].std()









##################
#Grid Search
##################

rf_model = RandomForestRegressor(random_state=17)
rf_model.get_params()

rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}

rf_best_grid = GridSearchCV(rf_model,
                            rf_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X_train, y_train)
rf_best_grid.best_params_

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(rf_final,
                                        X, y, cv=5, scoring="neg_mean_squared_error")))

print(rmse)


"""{'max_depth': None,
 'max_features': 'auto',
 'min_samples_split': 2,
 'n_estimators': 500}"""
################################
# Hyperparameter Optimization with RandomSearchCV
################################

rf_model =RandomForestRegressor (random_state=17)

rf_random_params = {"max_depth": np.random.randint(5, 50, 10),
                    "max_features": [3, 5, 7, "auto", "sqrt"],
                    "min_samples_split": np.random.randint(2, 50, 20),
                    "n_estimators": [int(x) for x in np.linspace(start=200, stop=1500, num=10)]}

rf_random = RandomizedSearchCV(estimator=rf_model,
                               param_distributions=rf_random_params,
                               n_iter=100,  # parameter num
                               cv=3,
                               verbose=True,
                               random_state=42,
                               n_jobs=-1)

rf_random.fit(X, y)


rf_random.best_params_


rf_random_final = rf_model.set_params(**rf_random.best_params_, random_state=17).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(rf_random_final,
                                        X, y, cv=5, scoring="neg_mean_squared_error")))

print(rmse)


