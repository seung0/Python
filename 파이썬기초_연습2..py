#!/usr/bin/env python
# coding: utf-8

# ## 과제를 위해 다음의 셀을 실행해 주세요.

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns

df = sns.load_dataset("titanic")
df.head()


# ## 1. 행과 열의 갯수를 출력해 주세요.
# 
# `(891, 15)` 라는 결과가 출력되도록 합니다.

# In[ ]:


df.shape


# ## 2. 결측치의 수를 세어주세요.
# * 아래의 결과가 출력되도록 합니다.
# ```
# survived         0
# pclass           0
# sex              0
# age            177
# sibsp            0
# parch            0
# fare             0
# embarked         2
# class            0
# who              0
# adult_male       0
# deck           688
# embark_town      2
# alive            0
# alone            0
# dtype: int64
# ```

# In[ ]:


df.isnull().sum()


# ## 3. alive 의 수를 그룹화해서 세어주세요.
# * 다음의 결과가 출력되도록 합니다.
# ```
# no     549
# yes    342
# Name: alive, dtype: int64
# ```

# In[ ]:


df['alive'].value_counts()


# ## 4. age의 요약값을 구합니다.
# * 다음의 결과가 출력되도록 합니다.
# ```
# count    714.000000
# mean      29.699118
# std       14.526497
# min        0.420000
# 25%       20.125000
# 50%       28.000000
# 75%       38.000000
# max       80.000000
# Name: age, dtype: float64
# ```

# In[ ]:


df['age'].describe()


# ## 5. object 타입의 요약값을 구합니다.
# * 모든 컬럼 중 object 타입인 컬럼만 요약합니다.
# * 요약 결과에 count, unique, top, freq 가 출력됩니다.

# In[ ]:


df[['sex','embarked','who','embark_town','alive']].describe()


# ## 6. embark_town 컬럼의 값을 소문자로 변경 후에 "embark_lower" 라는 새로운 컬럼에 담아주세요.
# * 그리고 embark_lower 컬럼에 담긴 값을 미리보기로 출력해 주세요. 아래의 결과가 표시되도록 합니다.
# 
# ```
# 0    southampton
# 1      cherbourg
# 2    southampton
# 3    southampton
# 4    southampton
# Name: embark_lower, dtype: object
# ```

# In[ ]:


df['embark_lower'] = df['embark_town'].str.lower()
df['embark_lower']


# ## 7. embark_lower 컬럼에서 south 가 들어가는 데이터의 수를 세어보세요.
# * True, False 로 출력되도록 하고 sum을 통해 값을 계산하면 644 가 출력됩니다.

# In[ ]:


df['embark_lower'].str.contains('south').sum()


# ## 8. age 컬럼의 값이 15 이하인 값을 구합니다.  True, False로 표시되는 값을 child 라는 새로운 컬럼에 담아주세요.

# In[ ]:


df['child'] = df['age'] <= 15
df['child']


# ## 9. 위에서 만든 child 컬럼을 통해 child 의 값을 그룹화 해서 세어봅니다.
# * 다음의 값이 출력되도록 합니다.
# ```
# False    808
# True      83
# Name: child, dtype: int64
# ```

# In[ ]:


df['child'].value_counts()


# ## 10. embarked 컬럼의 값이 C 이고 pclass 가 3에 해당되는 값만 가져와서 데이터프레임으로 만든 뒤 행과 열의 수를 세어주세요.
# * `(66, 17)` 라는 결과가 출력되도록 합니다.

# In[ ]:



df[(df['embarked'] == 'C') & (df['pclass'] == 3)].shape


# ## 11. fare 가 500보다 큰 값을 출력합니다.
# * 아래의 결과가 출력되도록 합니다.
# ```
# 258    512.3292
# 679    512.3292
# 737    512.3292
# Name: fare, dtype: float64
# ```

# In[ ]:


df['fare'][df['fare'] > 500]


# ## 12. pclass 가 3이고 embarked 가 Q인  fare 의 평균을 구해주세요.
# * `11.183393055555557` 라는 결과가 출력되도록 합니다.

# In[ ]:


df_1 = df[(df['pclass'] == 3) & (df['embarked'] == 'Q')]
df_1['fare'].mean()


# ## 13. fare 가 50 보다 큰 데이터에서 class 를 그룹화 해서 갯수를 세어보세요.
# * 다음의 결과가 출력되도록 합니다.
# ```
# First     139
# Third      14
# Second      7
# Name: class, dtype: int64
# ```

# In[ ]:


df_2 = df[df['fare'] > 50]
df_2.groupby(['class']).sum()


# ## 14. fare 가 10 보다 크고 50 미만인 데이터만 가져와서 데이터프레임의 크기를 출력해 주세요.
# * `(395, 17)`가 출력되도록 합니다.

# In[ ]:


df[(df['fare'] > 10) & (df['fare'] < 50)].shape


# ## 15. age 의 결측치를 0 으로 채워서 age_fill 이라는 컬럼에 담고 age_fill 컬럼의 평균값을 출력해 주세요.
# * `23.79929292929293` 이 출력되도록 합니다.

# In[ ]:


df['age_fill'] = df['age'].fillna(0)
df['age_fill'].mean()


# ## 16. deck 컬럼을 그룹화 해서 갯수를 카운트하고 A~G 순으로 정렬이 되도록 해주세요.
# * 다음의 결과가 출력되도록 합니다.
# ```
# A    15
# B    47
# C    59
# D    33
# E    32
# F    13
# G     4
# Name: deck, dtype: int64
# ```

# In[ ]:


df.groupby(['deck']).size()


# ## 17. pclass 컬럼의 값이 1 인 fare의 중앙값을 구해주세요.
# * `60.287499999999994` 라는 값이 출력되도록 합니다.

# In[ ]:


df.loc[df['pclass'] == 1, 'fare'].median()


# ## 18. embarked 가 C 이거나 deck 이 F인 데이터에서 age 컬럼의 평균값을 구해주세요.
# * `29.967517730496454` 값이 출력되도록 합니다.

# In[ ]:


df[(df['embarked'] == 'C') | (df['deck'] == 'F')]['age'].mean()


# ## 19. alive 가 yes 이고 alone 이 True인 값의 데이터프레임의 행과 열의 수를 출력해 주세요.
# * `(163, 18)` 라는 값이 출력되도록 합니다.

# In[ ]:


df[(df['alive'] == 'yes') & (df['alone'] == True)].shape


# ## 20. age 를 역순으로 정렬하고 상위 5개만 출력합니다.
# * 다음의 결과가 출력되도록 합니다.
# ```
# 630    80.0
# 851    74.0
# 96     71.0
# 493    71.0
# 116    70.5
# Name: age, dtype: float64
# ```

# In[ ]:


df['age'].sort_values(ascending = False).head()

