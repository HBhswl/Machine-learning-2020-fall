# sf-crime

### solution

#### time+address

| type                  | size  |
| --------------------- | ----- |
| address               | 2.3w+ |
| address - block       | 1.4w+ |
| address - /xx         | 1.2w+ |
| address - block - /xx | 2k+   |
|                       |       |

time 7-13 13-19 19-1 1-7

address - block - /xx



#### time+district

time 

work time（9-17）   else

work day （1-5）  else

district



### result



| default feature | address | time+place | time+district | params | score   |
| --------------- | ------- | ---------- | ------------- | ------ | ------- |
| yes             | no      | no         | no            | A      | 2.53492 |
| yes             | no      | no         | yes           | A      | 2.49411 |
| yes             | no      | yes        | no            | A      | 2.45616 |
| yes             | no      | yes        | yes           | A      | 2.51364 |
| yes             | yes     | no         | no            | A      | 2.66080 |
| yes             | yes     | no         | yes           | A      | 2.95312 |
| yes             | yes     | yes        | no            | A      | 2.71321 |
| + X Y + log     | yes     | no         | no            | B      | 2.22027 |
| + X Y + log     | yes     | no         | yes           | B      | 2.22344 |
|                 |         |            |               |        |         |
|                 |         |            |               |        |         |

Params A:

```
lgb_params = {    
	'boosting': 'gbdt',    
	'objective': 'multiclass',   
	'num_class': 39,    
	'max_delta_step': 0.9,    
	'min_data_in_leaf': 21,    
	'learning_rate': 0.4,   
	'max_bin': 465,   
 	'num_leaves': 41,
}

fold = 5
```



Params B:

```
lgb_params = {
    'num_leaves': 96,
    'min_data_in_leaf': 362, 
    'objective': 'multiclass',
    'num_classes': 39,
    'max_bin': 488,
    'learning_rate': 0.05686898284457517,
    'boosting': "gbdt",
    'metric': 'multi_logloss',
    'verbosity': 1,
    'num_round': 200,
    'silent': 0  
}

fold = 5
```



![](C:\Users\80592\AppData\Roaming\Typora\typora-user-images\1605772035624.png)



![1605771656168](C:\Users\80592\AppData\Roaming\Typora\typora-user-images\1605771656168.png)



default feature：

- Year
- Month
- Day
- Hour
- Minute
- Special Time
- Weekend
- Night
- DayOfWeek
- PdDistrict
- X
- Y



## How to deal address

#### coordinate 

ref: https://www.kaggle.com/dollardollar/importance-of-address-features

```python
# This provides us with a good idea as to which features are particularly relevant. 
# 
# - clearly, the timing in terms of minute, hour and year are critical
# - the collocated-crime feature scores surprisingly high
# - the spatial coordinates are useful
# - the total number of crimes in a steet is an important indicator, as well as some of the log-ratios
# - the month is not particularly essential, presumably as seasonal information can be recovered from the week
```



#### extract street suffix

ref: https://www.kaggle.com/espanarey/extracting-street-suffix-from-address

useful info https://pe.usps.com/text/pub28/28apc_002.htm

```
StreetAbbreviations <- data.frame(
Suffix = c("AV", "BL", "CR", "CT", "DR", "EX", "HWY", "HY", "LN", "PL", "PZ", "RD", "ST", "TR", "WY", "WAY"), 
SuffixName = c("Avenue", "Boulevard", "Circle", "Court", "Drive", "Expressway", "Highway", "Highway", "Lane", "Place", "Plaza", "Road", "Street", "Terrace", "Way", "Way"))

```



#### simplify

ref: https://benfradet.github.io/blog/2016/06/08/SF-crime-classification-with-Apache-Spark

##### Address features

Then, I wanted to make the `Address` variable usable. If you have a look at a few addresses in the dataset, you’ll notice that they come in two forms:

- {street 1} / {street 2} to denote an intersection
- {number} Block of {street}

Consequently, I introduced two features from this column:

- `AddressType` which indicates whether the incident took place at an intersection or on a particular street
- `Street` where I attempted to parse the `Address` variable to a single street name, this reduced the cardinality of the original feature by 10x

Unfortunately, the `Street` variable will only contain the first address (alphabetically) if `Address` is an intersection. So, is is possible that two addresses containing the same street represented by intersections won’t result in the same street.

For example, given two `Address`: `A STREET / B STREET` and `B STREET / C STREET` the resulting `Street` will be `A STREET` and `B STREET`.





#### division

1. address embedding + intersection
2.  StreetNo and Block
3. street + intersection

