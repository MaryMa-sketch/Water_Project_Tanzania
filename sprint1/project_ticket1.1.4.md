# **Initial observations and questions for further investigation**

## Dataset Overview

The project has 4 datasets:
     - SubmissionFormat
     - Test set
     - Training set 
     - Training set labels(target variables)

- **Training_set** has 59400 rows and 40 feature columns
- **Test_set** has 14850 rows and 40 feature columns
- **Training set lables** - has Target variables(status_group) was merged to the Training_set
- **Features:**
   - **Categorical columns**: 'funder', 'installer', 'wpt_name', 'basin', 'lga', 'ward', 'public_meeting', 'recorded_by','scheme_management', 'scheme_name', 'permit', 'extraction_type', 'extraction_type_group', 'extraction_type_class', 'management',
    'management_group', 'payment', 'payment_type', 'water_quality', 'quality_group', 'quantity', 'quantity_group', 'source', 'source_type', 'source_class', 'waterpoint_type', 'waterpoint_type_group'
   - **Numerical columns**: 'amount_tsh', 'gps_height', 'longitude', 'latitude',
       'num_private', 'region_code', 'district_code', 'population', 'construction_year'
   - **Temporal and location**: 'date_recorded', subvillage, region, ward, construction year

**Data challenges**             
- **missing values** - scheme_name, scheme_management,installer, funder, public_meeting, permit, subvillage. scheme_name has almost 50% missing data.
- **class imbalance** - target distribution is skewed towards functional..e.g need repair is only 7%
- **incorrect entries** - 0 entries for construction year, population, Gps_height,longitude
- **Overlapping columns** :- extraction_type, Eéxtraction_type_group, extraction_type_class
                         - payment, payment_type
                         - water quality and quality group
                         - quantity and quantity_group
                         - source, source_type and source_class
                         - water_type and water_type_group
                          
**Outliers** - amount_tsh, population
**Abbrevations** - wpt_name, Iga 
**Inconsistent category names** (spelling, case variables)- Taasaf and Tasaf, German republi and German republic, danida and danid, District counsil and District Counsil, 



# **Questions for analysis**
 - are zeros in population, construction year, longitude, amount_tsh truly zero or missing values?
 - should the overlapping columns be combined to one ?
 - how should we handle class imbalance? class weights vs oversampling
 - does functionalty vary by installer, extraction_type, basin or region, waterpoint_type, management?
 - what feature engineering can be done?

