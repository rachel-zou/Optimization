from pyspark.sql.functions import *
import pulp
import pandas as pd

# For VIP, assume we will see 20% pull forward during the two promotions weeks and 10% decrease in the previous and following weeks 
df = spark.sql('''select * from sandbox.rz_canton_pro_optimize''')\
          .withColumn('week_rate', when((col('week_id')==3.0)|(col('week_id')==7.0),1.2)\
                                  .when((col('week_id')==2.0)|(col('week_id')==4.0)|(col('week_id')==6.0)|(col('week_id')==8.0),0.9)\
                                  .otherwise(1))

hh_df = df.toPandas()

pro_list = hh_df['hshld_no'].unique().tolist()
low_risk_list = hh_df[hh_df['perk_value']<0.05]['hshld_no'].unique().tolist()

vip_fg = pulp.LpVariable.dicts("vip_fg", ((hshld_no) for hshld_no in pro_list), cat='Binary')
non_vip_fg = pulp.LpVariable.dicts("non_vip_fg", ((hshld_no) for hshld_no in pro_list), cat='Binary')

hh_df['vip_fg'] = hh_df['hshld_no'].map(vip_fg)
hh_df['non_vip_fg'] = hh_df['hshld_no'].map(non_vip_fg)

# Margin rates
ge_rate = 0.35
cs_rate = 0.35
gallon_rate = 0.22
rx_rate = 0.20
gc_rate = 0.067

# State problem: Maximize margin
model = pulp.LpProblem("margin_maximizing_problem", pulp.LpMaximize)

# Objective
model += pulp.lpSum( hh_df['non_ob_spend']*hh_df['week_rate']*hh_df['vip_fg']*ge_rate ) + pulp.lpSum( hh_df['non_ob_spend']*hh_df['non_vip_fg']*ge_rate ) \
        + pulp.lpSum( hh_df['ob_spend']*hh_df['week_rate']*hh_df['vip_fg']*ge_rate ) + pulp.lpSum( hh_df['ob_spend']*hh_df['non_vip_fg']*ge_rate ) \
        + pulp.lpSum( hh_df['fuel_gallons']*hh_df['week_rate']*hh_df['vip_fg']*gallon_rate ) + pulp.lpSum( hh_df['fuel_gallons']*hh_df['non_vip_fg']*gallon_rate ) \
        + pulp.lpSum( hh_df['gc_spend']*hh_df['week_rate']*hh_df['vip_fg']*gc_rate ) + pulp.lpSum( hh_df['gc_spend']*hh_df['non_vip_fg']*gc_rate ) \
        + pulp.lpSum( hh_df['cs_spend']*hh_df['week_rate']*hh_df['vip_fg']*cs_rate ) + pulp.lpSum( hh_df['cs_spend']*hh_df['non_vip_fg']*cs_rate ) \
        - pulp.lpSum( hh_df['ob_spend']*hh_df['week_rate']*hh_df['vip_fg']*0.05 ) \
        - pulp.lpSum( (hh_df['non_ob_spend']*hh_df['week_rate']*hh_df['vip_fg']*1.5)/50 ) - pulp.lpSum( (hh_df['non_ob_spend']*hh_df['non_vip_fg'])/50 ) \
        - pulp.lpSum( (hh_df['ob_spend']*hh_df['week_rate']*hh_df['vip_fg']*1.5)/50 ) - pulp.lpSum( (hh_df['ob_spend']*hh_df['non_vip_fg'])/50 ) \
        - pulp.lpSum( (hh_df['fuel_gallons']*hh_df['week_rate']*hh_df['vip_fg']*3)/50 ) - pulp.lpSum( (hh_df['fuel_gallons']*hh_df['non_vip_fg']*2)/50 ) \
        - pulp.lpSum( (hh_df['gc_spend']*hh_df['week_rate']*hh_df['vip_fg']*1.5)/50 ) - pulp.lpSum( (hh_df['gc_spend']*hh_df['non_vip_fg'])/50 ) \
        - pulp.lpSum( (hh_df['cs_spend']*hh_df['week_rate']*hh_df['vip_fg']*1.5)/50 ) - pulp.lpSum( (hh_df['cs_spend']*hh_df['non_vip_fg'])/50 ) 

# Constraints
# The number of total VIPs 
model += pulp.lpSum(vip_fg.values()) <= 6000

# Each HH is either VIP or non VIP
for hh in pro_list:
  model += vip_fg[hh] + non_vip_fg[hh] == 1, "VIP_Status_%s"%hh
  
# Only select from high risk perk value segment
for hh in low_risk_list:
  model += vip_fg[hh] == 0, "Low_Risk_VIP_Status_%s"%hh

model.solve()
pulp.LpStatus[model.status]

# The maximum margin
print("objective=", pulp.value(model.objective))

# Store the results in a pyspark dataframe
output = []

for hshld_no in vip_fg:
  var_output = {
    'hshld_no' : hshld_no,
    'vip_fg': vip_fg[(hshld_no)].varValue
  }
  output.append(var_output)
  
output_df = pd.DataFrame.from_records(output)

schema = StructType([
  StructField('hshld_no', StringType(), True),
  StructField('vip_fg', DoubleType(), True),
  ])

vip_df = spark.createDataFrame(output_df, schema)

vip_df.write.mode('overwrite').saveAsTable(name='sandbox.rz_pulp_vip',format='delta')