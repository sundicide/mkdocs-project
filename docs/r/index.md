# CSV
## Read CSV
```r
my_csv = read.csv('my.csv', row.names = 1, stringsAsFactors = FALSE)
```
## Select Some Columns
```r
# select all rows and select some columns
# (col1, col2, col3)
my_csv = my_csv[, c('col1', 'col2', 'col3')]
```


# DataFrame
## Create DataFrame
```r
df <- c(1,2,3,4)
랭크 <- c('D', 'E', 'A', 'B')
data.frame(df, 랭크)
```

## row 수를 알아내는 방법
```r
nrow(df)
```

## bind row names to column values
```r
# my_csv의 rownames를 컬럼 값으로 넣는다.
rownames(df) = df[['랭크']]
#  age   name 랭크
# D  25    Tom    D
# E  34  Harry    E
# A  28 Porter    A
# B  52 Harden    B
```

컬럼 값을 이용하는 대신 pad도 이용한다.
예를 들어 value2안에 code 컬럼의 값이 0001, 00002, 000003 라고 하자.
이럴 때 아래와 같이 해주면 rownames에는 다음과 같이 채워진다.
> 000001
000002
000003

str_pad를 이용해 자리수가 부족한 만큼 0으로 채워주는 것이다
```r
rownames(value2) = str_pad(value2$'code', 6, side=c('left'), pad='0')
```


## transpose df
dataframe의 행과 열을 바꿀때 사용한다.
```r
df %>% t()
```

## get last rows of df
df내에서 마지막 줄 만 얻고 싶을 때 사용한다.
```r
rowLength = nrow(df)
df[rowLength, ]
```

## change column name
컬럼 명을 원하는 이름으로 바꿀 수 있다.
```r
df = df[, c('컬럼1', '컬럼2')]
```

또는 column이 1개일 경우 아래와 같이 할 수도 있다.
```r
colnames(df) = '컬럼'
```

dff에는 dataframe 중첩으로 들어가 있다.
df내 다시 df
그래서 구하고자 하는 값은
dff의 컬럼명이 '컬럼1'인 값들인 '컬럼2'인 값들을 곱한뒤 2번 째 컬럼에 해당하는 값을 뽑고 그 컬럼명을 'GG'라고 하고 싶을때 아래와 같이 할 수 있다.
```r
data_gpa =
  (dff$'컬럼1' * dff$'컬럼2')[2] %>%
    setNames('GG')
```


## get data
```r
df[[1]] # 첫 번째 행 값들을 얻는다.
df[1] # c첫 번째 열 값들을 얻는다.

df$'컬럼' # 컬럼명이 '컬럼'인 열 값들을 얻는다.

df['행1', '컬럼1'] # 행명이 '행1'이고 컬럼명이 '컬럼1'인 데이터를 얻는다.
```

## set data
```r
myData[['id']] = 'myId'
myData[['name']] = 'you'
df[[i]] = myData
```



## 특정 값에 해당하는 컬럼 index 찾기
컬럼 이름 중 작년에 해당하는 컬럼 index 찾기
```r
num_col = str_which(colnames(df), as.character(lubridate::year(Sys.Date()) - 1))
```

## for문
```r
for (i in 1 : nrow(df)) {
  curr = data[[i]]
}
```

## bind_rows
dplyr 패키지내 bind_rows를 사용하면 행 기준으로 합칠 수 있다.
열이 비어있으면 자동적으로 NA를 삽입한다.
```r
library(dplyr)
# > making 2 data.frame examples
# > df_1 <- data.frame(x = 1:3, y = 1:3)
# > df_2 <- data.frame(x = 4:6, y = 4:6)

df_1 <- data.frame(x = 1:3, y = 1:3)
df_2 <- data.frame(x = 4:6, y = 4:6)

bind_rows(df_1, df_2)
# x y
# 1 1 1
# 2 2 2
# 3 3 3
# 4 4 4
# 5 5 5
# 6 6 6

```

## order by column data
```r
# 컬럼1 데이터로 DESC 정렬
# ASC 정렬을 할때는 - 를 제거하면 된다.
# 컬럼1, 컬럼2로 정렬하고 싶다면 order안에 파라미터로 추가하면 된다.
orderedSector = df[order(-df$'컬럼1'),]
```




# Lubidrate
```r
library(lubridate)
```

## get now month
```r
lubridate::month(Sys.Date())
```

# Strings

## substring

```r
raw = 'X123456'
substr(raw, 2, 7) # 인덱스가 1부터 해서 7번째에서 멈춘다.
# 123456

substr(raw, 1, 1)
# X

substr(raw, 0, 1)
# X

substr(raw, 0, 2)
# X1

substr(raw, 0, 10)
# X123456
```

## get length
```r
nchar(myStr)
```

