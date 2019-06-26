#---------------------------------------------------------------------------------------------------------------------------
#
# ZMUM
# Projekt nr 1
#
#---------------------------------------------------------------------------------------------------------------------------
#
# Kod
#
#---------------------------------------------------------------------------------------------------------------------------

### Wczytanie pakietow

library(e1071)
library(caret)
library(MLmetrics)
library(adabag)
library(fastAdaboost)
library(randomForest)
library(xgboost)
library(rpart)
library(rpart.plot)

#---------------------------------------------------------------------------------------------------------------------------

### Funkcje pomocnicze

## Funkcja do obliczania precyzji @ k%
prec_k<- function(y_prob, y_true, k) {
  #y_pred <- ifelse(y_prob >= quantile(y_prob, 1 - k), 1, 0)
  ind <- order(y_prob, decreasing = TRUE)[1:ceiling(length(y_prob) * k)]
  y_pred <- numeric(length(y_prob))
  y_pred[ind] <- 1
  tab <- table(y_true, y_pred)
  precision <- tab[2,2] / sum(tab[,2])
  precision
}

#---------------------------------------------------------------------------------------------------------------------------

### Wczytanie danych

train <- read.table("https://home.ipipan.waw.pl/p.teisseyre/TEACHING/ZMUM/DANE/Projekt1/train.txt", header = TRUE)
head(train)
nrow(train)   # 40000
ncol(train)   # 231

testx <- read.table("https://home.ipipan.waw.pl/p.teisseyre/TEACHING/ZMUM/DANE/Projekt1/testx.txt", header = TRUE)
head(testx)
nrow(testx)    # 10000
ncol(testx)    # 230

#---------------------------------------------------------------------------------------------------------------------------

### Przeglad danych

nrow(train[train$class == 1,]) / nrow(train)   # ~ 7% obserwacji nalezy do klasy 1 => klasy sa niezrownowazone

#---------------------------------------------------------------------------------------------------------------------------

### Oczyszczenie danych


### 1) Obsluga brakow danych -----------------------------------------------------------------------------------------------


## Zliczenie liczby NA

# train
sum(is.na(train))   # 6102025

# testx
sum(is.na(testx))    # 1525934


## Sprawdzenie procentu liczby NA z podzialem na klasy
class1_col_na <- colSums(is.na(train[train$class == 1,])) / nrow(train[train$class == 1,])
class0_col_na <- colSums(is.na(train[train$class == 0,])) / nrow(train[train$class == 0,])
col_na <- colSums(is.na(train)) / nrow(train)
class_col_na_df <- data.frame(class0_col_na, class1_col_na, col_na)
class_col_na_df$diff <- abs(class0_col_na - class1_col_na)
max(class_col_na_df$diff)   # 0.06742331

# Wniosek: zarowno w obserwacjach klasy 1 jak i klasy 0 jest podobny procent NA w poszczegolnych zmiennych


## Zliczenie liczby NA w kazdej zmiennej

# train
train_na_count <- colSums(is.na(train)) / nrow(train)
train_na_col_count <- sum(train_na_count > 0.97)
train_na_col_names <- names(train_na_count[train_na_count > 0.97])

# testx
testx_na_count <- colSums(is.na(testx)) / nrow(testx)
testx_na_col_count <- sum(testx_na_count > 0.97)
testx_na_col_names <- names(testx_na_count[testx_na_count > 0.97])

# Porownanie train i testx
train_na_col_count == testx_na_col_count   # FALSE
train_na_col_count  # 106
testx_na_col_count  # 128

# Wniosek: w train jest mniej zmiennych zawierajacych ponad 97% NA niz w testx, 
# ale pozbywam sie zmiennych, ktore zawieraja ponad 97% NA w train,
# zeby w train i testx pozostaly te same kolumny
col_names_to_cut <- names(train_na_count[train_na_count > 0.97])


## Wyciecie zmiennych zawierajacych same NA

# train
train <- train[(colnames(train) %in% col_names_to_cut) == FALSE]
ncol(train)     # 125

# testx
testx <- testx[(colnames(testx) %in% col_names_to_cut) == FALSE]
ncol(testx)      # 124


## Zliczenie liczby NA w kazdej obserwacji

# train
train_na_count <- rowSums(is.na(train)) / ncol(train)
train_na_col_count <- sum(train_na_count == 1)
train_na_col_count  # 0

# testx
testx_na_count <- rowSums(is.na(testx)) / ncol(testx)
testx_na_col_count <- sum(testx_na_count == 1)
testx_na_col_count  # 0

# Wniosek: nie ma obserwacji zawierajacych same NA


## Rozdzielenie zmiennych ilosciowych i jakosciowych
train_types_of_columns <- sapply(train, function(x) {class(x)})
types_of_columns <- unique(train_types_of_columns)    # "integer" "numeric" "factor" 
train_factor_col_names <- colnames(train[train_types_of_columns == "factor"])
train_numeric_col_names <- colnames(train[train_types_of_columns != "factor"])


## Zmienne ilosciowe
head(train[colnames(train) %in% train_numeric_col_names])
head(testx[colnames(testx) %in% train_numeric_col_names])

# Zastapienie NA medianami za pomoca funkcji impute z pakietu e1071
train[colnames(train) %in% train_numeric_col_names] <- impute(train[colnames(train) %in% train_numeric_col_names], "median")
testx[colnames(testx) %in% train_numeric_col_names] <- impute(testx[colnames(testx) %in% train_numeric_col_names], "median")


### 2) Obsluga zmiennych jakosciowych ---------------------------------------------------------------------------------------


## Zliczenie liczby unikalnych wartosci w kazdej kolumnie

# train
unique_col_values_train <- sapply(train[colnames(train) %in% train_factor_col_names], function(x) {length(unique(x))})
unique_col_values_train_col_names <- names(unique_col_values_train[unique_col_values_train > 51])
sum(unique_col_values_train == nrow(train))   # 0 => zadna kolumna nie ma tyle unikalnych wartosci co wierszy

# testx
unique_col_values_testx <- sapply(testx[colnames(testx) %in% train_factor_col_names], function(x) {length(unique(x))})
unique_col_values_testx_col_names <- names(unique_col_values_testx[unique_col_values_testx > 51])
sum(unique_col_values_testx == nrow(testx))   # 0 => zadna kolumna nie ma tyle unikalnych wartosci co wierszy

# Porównanie train i testx
sum(unique_col_values_train_col_names == unique_col_values_testx_col_names) / length(unique_col_values_train_col_names)

# Wniosek: zarowno w train jak i w testx te same zmienne zawieraja ponad 51 roznych kategorii - pozbywam sie tych zmiennych
col_names_to_cut <- names(unique_col_values_train[unique_col_values_train > 51])


## Wyciecie zmiennych zawierajacych ponad 51 roznych kategorii

# train
train <- train[(colnames(train) %in% col_names_to_cut) == FALSE]
ncol(train)     # 112

# testx
testx <- testx[(colnames(testx) %in% col_names_to_cut) == FALSE]
ncol(testx)      # 111


#---------------------------------------------------------------------------------------------------------------------------

### Stworzenie liczbowego odpowiednika danych train - zamiana kolumn tekstowych na liczbowe

train_numeric <- train

# Aktualizacja zmiennej train_factor_col_names
train_types_of_columns <- sapply(train, function(x) {class(x)})
train_factor_col_names <- colnames(train[train_types_of_columns == "factor"])

# Zmiana factorow na wartosci liczbowe
train_numeric[colnames(train_numeric) %in% train_factor_col_names] <- sapply(train_numeric[colnames(train_numeric) %in% train_factor_col_names], function(x) {as.numeric(x)})


#---------------------------------------------------------------------------------------------------------------------------

### Podzial zbioru danych train na zbior treningowy i testowy

## Zbior train

train$class <- as.factor(train$class)

trainIndex <- createDataPartition(train$class, p = 0.9, list = FALSE)
data_train <- train[trainIndex,]
data_test <- train[-trainIndex,]

head(data_train)
nrow(data_train)

head(data_test)
nrow(data_test)

nrow(data_train[data_train$class == 1,]) / nrow(data_train)
nrow(data_test[data_test$class == 1,]) / nrow(data_test)
# nadal ~7% obserwacji nalezy do klasy 1 (w obu zbiorach)


## Zbior train_numeric

train_numeric_Index <- createDataPartition(train_numeric$class, p = 0.9, list = FALSE)
data_train_numeric <- train_numeric[train_numeric_Index,]
data_test_numeric <- train_numeric[-train_numeric_Index,]

head(data_train_numeric)
nrow(data_train_numeric)

head(data_test_numeric)
nrow(data_test_numeric)

nrow(data_train_numeric[data_train_numeric$class == 1,]) / nrow(data_train_numeric)
nrow(data_test_numeric[data_test_numeric$class == 1,]) / nrow(data_test_numeric)
# nadal ~7% obserwacji nalezy do klasy 1 (w obu zbiorach)


#---------------------------------------------------------------------------------------------------------------------------

### Budowanie modeli + predykcja + precyzja @ 10% - pierwsze, pojedyncze podejscie, domyslne parametry

## pojedyncze drzewo -------------------------------------------------------------------------------------------------------

treeModel <- rpart(class ~ ., data_train)
treeProb <- predict(treeModel, newdata = data_test[,-ncol(data_test)], type = "prob")[,2]
treePred <- predict(treeModel, newdata = data_test[,-ncol(data_test)], type = "class")

Precision(y_pred = treePred, y_true = data_test$class)          # 0.9602718 ; 0.9602926 ; 0.9568719 ; 0.9595195
AUC(y_pred = treePred, y_true = data_test$class)                # 0.736297  ; 0.7365668 ; 0.7136205 ; 0.7311776
prec_k(y_prob = treeProb, y_true = data_test$class, k = 0.1)    # 0.6135458 ; 0.6156863 ; 0.602459  ; 0.5884615

pdf(file="treeModel.pdf")
rpart.plot(treeModel)
dev.off()


treeModel <- rpart(class ~ ., data_train, cp = 0.001, minsplit = 5)
treeProb <- predict(treeModel, newdata = data_test[,-ncol(data_test)], type = "prob")[,2]
treePred <- predict(treeModel, newdata = data_test[,-ncol(data_test)], type = "class")

Precision(y_pred = treePred, y_true = data_test$class)          # 0.957076  ; 0.9577575 ; 0.9546867 ; 0.958062
AUC(y_pred = treePred, y_true = data_test$class)                # 0.7147873 ; 0.7190972 ; 0.6988018 ; 0.7214783
prec_k(y_prob = treeProb, y_true = data_test$class, k = 0.1)    # 0.6122449 ; 0.6166008 ; 0.6111111 ; 0.59375

pdf(file="treeModel.pdf")
rpart.plot(treeModel)
dev.off()


## adaboost ----------------------------------------------------------------------------------------------------------------

# adaboost z pakietu adabag
# dziala bardzo wolno (ponizszy trening robil sie ok 4,5 h)

adaboostModel <- boosting(class ~ ., data_train)
adaboostProb <- predict(adaboostModel, newdata = data_test[,-ncol(data_test)])$prob[,2]
adaboostPred <- predict(adaboostModel, newdata = data_test[,-ncol(data_test)])$class

Precision(y_pred = adaboostPred, y_true = data_test$class)          # 0.9530845
AUC(y_pred = adaboostPred, y_true = data_test$class)                # 0.6872137
prec_k(y_prob = adaboostProb, y_true = data_test$class, k = 0.1)    # 0.36


# adaboost z pakietu fastAdaboost
# dziala szybciej (ponizszy trening robil sie ok 0,5 h)

adaboostModel2 <- adaboost(class ~ ., data_train, nIter = 100)
adaboostProb2 <- predict(adaboostModel2, newdata = data_test[,-ncol(data_test)])$prob[,2]
adaboostPred2 <- predict(adaboostModel2, newdata = data_test[,-ncol(data_test)])$class

Precision(y_pred = adaboostPred2, y_true = data_test$class)          # 0.9550299
AUC(y_pred = adaboostPred2, y_true = data_test$class)                # 0.7004608
prec_k(y_prob = adaboostProb2, y_true = data_test$class, k = 0.1)    # 0.3825


# W zwiazku z tym, ze adaboost z pakietu adabag dziala o wiele wolniej niz z pakietu fastAdaboost, do testow uzyje adaboost
# z pakietu fastAdaboost


## random forest -----------------------------------------------------------------------------------------------------------

randomForestModel <- randomForest(class ~ ., data_train, ntree = 100)
randomForestProb <- predict(randomForestModel, newdata = data_test[,-ncol(data_test)], type = 'prob')[,2]
randomForestPred <- ifelse(randomForestProb < 0.5, 0, 1)

Precision(y_pred = randomForestPred, y_true = data_test$class)          # 0.9509171
AUC(y_pred = randomForestPred, y_true = data_test$class)                # 0.672395
prec_k(y_prob = randomForestProb, y_true = data_test$class, k = 0.1)    # 0.3551637


## xgboost -----------------------------------------------------------------------------------------------------------------

dtrain <- xgb.DMatrix(data = data.matrix(data_train_numeric[,-ncol(data_train_numeric)]), label= data_train_numeric$class)
dtest <- xgb.DMatrix(data = data.matrix(data_test_numeric[,-ncol(data_test_numeric)]), label= data_test_numeric$class)


xgboostModel <- xgboost(data = dtrain, nrounds = 2, num_class = 2, objective = "multi:softprob")
xgboostProb <- predict(xgboostModel, dtest)
xgboostProb <- matrix(xgboostProb, ncol = 2, byrow = T)[,2]
xgboostPred <- ifelse(xgboostProb < 0.5, 0, 1)

Precision(y_pred = xgboostPred, y_true = data_test_numeric$class)          # 0.9534702 ; 0.9578125 ; 0.9637643
AUC(y_pred = xgboostPred, y_true = data_test_numeric$class)                # 0.7061653 ; 0.7152769 ; 0.7234272
prec_k(y_prob = xgboostProb, y_true = data_test_numeric$class, k = 0.1)    # 0.4235925 ; 0.4273743 ; 0.4294872


xgboostModel <- xgboost(data = dtrain, nrounds = 10, num_class = 2, objective = "multi:softprob")
xgboostProb <- predict(xgboostModel, dtest)
xgboostProb <- matrix(xgboostProb, ncol = 2, byrow = T)[,2]
xgboostPred <- ifelse(xgboostProb < 0.5, 0, 1)

Precision(y_pred = xgboostPred, y_true = data_test_numeric$class)          # 0.960499  ; 0.9583116 ; 0.9630977
AUC(y_pred = xgboostPred, y_true = data_test_numeric$class)                # 0.7253456 ; 0.7187372 ; 0.7187928
prec_k(y_prob = xgboostProb, y_true = data_test_numeric$class, k = 0.1)    # 0.4055416 ; 0.38      ; 0.3693931


# Maly test parametru nrounds (jednorazowe testy dla 4 roznych wartosci parametru nrounds, dwa razy zrobiony 
# podzial kroswalidacyjny na zbior treningowy i testowy)

dtrain <- xgb.DMatrix(data = data.matrix(data_train_numeric[,-ncol(data_train_numeric)]), label= data_train_numeric$class)
dtest <- xgb.DMatrix(data = data.matrix(data_test_numeric[,-ncol(data_test_numeric)]), label= data_test_numeric$class)


# nrounds = 2
xgboostModel <- xgboost(data = dtrain, nrounds = 2, num_class = 2, objective = "multi:softprob")
xgboostProb <- predict(xgboostModel, dtest)
xgboostProb <- matrix(xgboostProb, ncol = 2, byrow = T)[,2]
xgboostPred <- ifelse(xgboostProb < 0.5, 0, 1)

Precision(y_pred = xgboostPred, y_true = data_test_numeric$class)          # 0.9592689 ; 0.9579963
AUC(y_pred = xgboostPred, y_true = data_test_numeric$class)                # 0.733458  ; 0.7117565
prec_k(y_prob = xgboostProb, y_true = data_test_numeric$class, k = 0.1)    # 0.4736842 ; 0.4285714


# nrounds = 10
xgboostModel <- xgboost(data = dtrain, nrounds = 10, num_class = 2, objective = "multi:softprob")
xgboostProb <- predict(xgboostModel, dtest)
xgboostProb <- matrix(xgboostProb, ncol = 2, byrow = T)[,2]
xgboostPred <- ifelse(xgboostProb < 0.5, 0, 1)

Precision(y_pred = xgboostPred, y_true = data_test_numeric$class)          # 0.9592583 ; 0.9585073
AUC(y_pred = xgboostPred, y_true = data_test_numeric$class)                # 0.733323  ; 0.7153999
prec_k(y_prob = xgboostProb, y_true = data_test_numeric$class, k = 0.1)    # 0.4435897 ; 0.3775


# nrounds = 50
xgboostModel <- xgboost(data = dtrain, nrounds = 50, num_class = 2, objective = "multi:softprob")
xgboostProb <- predict(xgboostModel, dtest)
xgboostProb <- matrix(xgboostProb, ncol = 2, byrow = T)[,2]
xgboostPred <- ifelse(xgboostProb < 0.5, 0, 1)

Precision(y_pred = xgboostPred, y_true = data_test_numeric$class)          # 0.9580183 ; 0.9570201
AUC(y_pred = xgboostPred, y_true = data_test_numeric$class)                # 0.7250405 ; 0.7050081
prec_k(y_prob = xgboostProb, y_true = data_test_numeric$class, k = 0.1)    # 0.43      ; 0.3675


# nrounds = 100
xgboostModel <- xgboost(data = dtrain, nrounds = 100, num_class = 2, objective = "multi:softprob")
xgboostProb <- predict(xgboostModel, dtest)
xgboostProb <- matrix(xgboostProb, ncol = 2, byrow = T)[,2]
xgboostPred <- ifelse(xgboostProb < 0.5, 0, 1)

Precision(y_pred = xgboostPred, y_true = data_test_numeric$class)          # 0.9570424 ; 0.9570312
AUC(y_pred = xgboostPred, y_true = data_test_numeric$class)                # 0.7185765 ; 0.7051427
prec_k(y_prob = xgboostProb, y_true = data_test_numeric$class, k = 0.1)    # 0.4325    ; 0.365


# Wniosek: im wieksza wartosc parametru nrounds, tym model daje gorsze wyniki (wg miary prec10%). Ponadto dla duzych 
# wartosci tego parametru komputer wykorzystuje 100% mocy procesora. Zatem dokladniejsze testy przeprowadze dla
# malych wartosci parametru nrounds.


#---------------------------------------------------------------------------------------------------------------------------

### Testowanie wybranych klasyfikatoroW

## pojedyncze drzewo -----------------------------------------------------------------------------------------------------------------

# Test nr 1 - parametr minsplits
number_of_iterations <- 10
minsplits <- seq(10, 100, 10)
precision_tree <- auc_tree <- precision_10_tree <- data.frame(matrix(0, nrow = number_of_iterations, ncol = length(minsplits)))

for(i in 1:number_of_iterations) {
  # podział na train i test
  trainIndex <- createDataPartition(train$class, p = 0.9, list = FALSE)
  data_train <- train[trainIndex,]
  data_test <- train[-trainIndex,]
  
  # uczenie modelu + predict na zbiorze testowym
  for(j in 1:length(minsplits)) {
    treeModel <- rpart(class ~ ., data_train, minsplit = minsplits[j])
    treeProb <- predict(treeModel, newdata = data_test[,-ncol(data_test)], type = "prob")[,2]
    treePred <- predict(treeModel, newdata = data_test[,-ncol(data_test)], type = "class")
    
    precision_tree[i, j] <- Precision(y_pred = treePred, y_true = data_test$class)
    auc_tree[i, j] <- AUC(y_pred = treePred, y_true = data_test$class)
    precision_10_tree[i, j] <- prec_k(y_prob = treeProb, y_true = data_test$class, k = 0.1)
  }
}

# Wyznaczenie srednich wartosci miar dla poszczegolnych modeli
colMeans(precision_tree)     # 0.9574492 0.9574492 0.9574492 0.9574492 0.9574492 0.9574492 0.9574492 0.9574492 0.9574492 0.9574492 
colMeans(auc_tree)           # 0.7170407 0.7170407 0.7170407 0.7170407 0.7170407 0.7170407 0.7170407 0.7170407 0.7170407 0.7170407
colMeans(precision_10_tree)  # 0.5948064 0.5948064 0.5948064 0.5948064 0.5948064 0.5948064 0.5948064 0.5948064 0.5948064 0.5948064

# Niezaleznie od wartosci parametru minsplit wyniki wyszly takie same. Sprawdze, co mi dadza wieksze wartosci tego parametru. 


# Test nr 2 - parametr minsplits
number_of_iterations <- 10
minsplits <- seq(100, 1000, 100)
precision_tree <- auc_tree <- precision_10_tree <- data.frame(matrix(0, nrow = number_of_iterations, ncol = length(minsplits)))

for(i in 1:number_of_iterations) {
  # podział na train i test
  trainIndex <- createDataPartition(train$class, p = 0.9, list = FALSE)
  data_train <- train[trainIndex,]
  data_test <- train[-trainIndex,]
  
  # uczenie modelu + predict na zbiorze testowym
  for(j in 1:length(minsplits)) {
    treeModel <- rpart(class ~ ., data_train, minsplit = minsplits[j])
    treeProb <- predict(treeModel, newdata = data_test[,-ncol(data_test)], type = "prob")[,2]
    treePred <- predict(treeModel, newdata = data_test[,-ncol(data_test)], type = "class")
    
    precision_tree[i, j] <- Precision(y_pred = treePred, y_true = data_test$class)
    auc_tree[i, j] <- AUC(y_pred = treePred, y_true = data_test$class)
    precision_10_tree[i, j] <- prec_k(y_prob = treeProb, y_true = data_test$class, k = 0.1)
  }
}

# Wyznaczenie srednich wartosci miar dla poszczegolnych modeli
colMeans(precision_tree)     # 0.9560712 0.9564133 0.9564133 0.9564133 0.9564133 0.9564133 0.9564133 0.9564133 0.9564133 0.9564133
colMeans(auc_tree)           # 0.7073670 0.7094228 0.7094228 0.7094228 0.7094228 0.7094228 0.7094228 0.7094228 0.7094228 0.7094228
colMeans(precision_10_tree)  # 0.5782629 0.5782629 0.5782629 0.5782629 0.5782629 0.5782629 0.5782629 0.5782629 0.5782629 0.5782629

# W przypadku tych danych parametr minsplit nie wpływa na model (wg miary prec10%).


# Test nr 3 - parametr cp
number_of_iterations <- 10
cps <- c(0.001, 0.01, 0.1)
precision_tree <- auc_tree <- precision_10_tree <- data.frame(matrix(0, nrow = number_of_iterations, ncol = length(cps)))

for(i in 1:number_of_iterations) {
  # podział na train i test
  trainIndex <- createDataPartition(train$class, p = 0.9, list = FALSE)
  data_train <- train[trainIndex,]
  data_test <- train[-trainIndex,]
  
  # uczenie modelu + predict na zbiorze testowym
  for(j in 1:length(cps)) {
    treeModel <- rpart(class ~ ., data_train, cp = cps[j])
    treeProb <- predict(treeModel, newdata = data_test[,-ncol(data_test)], type = "prob")[,2]
    treePred <- predict(treeModel, newdata = data_test[,-ncol(data_test)], type = "class")
    
    precision_tree[i, j] <- Precision(y_pred = treePred, y_true = data_test$class)
    auc_tree[i, j] <- AUC(y_pred = treePred, y_true = data_test$class)
    precision_10_tree[i, j] <- prec_k(y_prob = treeProb, y_true = data_test$class, k = 0.1)
  }
}

# Wyznaczenie srednich wartosci miar dla poszczegolnych modeli
colMeans(precision_tree)     # 0.9561390 0.9569097 0.9571978
colMeans(auc_tree)           # 0.7080638 0.7133404 0.7150009
colMeans(precision_10_tree)  # 0.5941844 0.5930246 0.5930246

# Wyznaczenie modelu, dla ktorego wyszla najwieksza precyzja @ 10%
which.max(colMeans(precision_10_tree))  # model nr 1  => cp = 0.001
max(colMeans(precision_10_tree))        # 0.5941844


# Test nr 4 - parametr cp - mniejsze wartosci
number_of_iterations <- 10
cps <- seq(0.001, 0.01, 0.001)
precision_tree <- auc_tree <- precision_10_tree <- data.frame(matrix(0, nrow = number_of_iterations, ncol = length(cps)))

for(i in 1:number_of_iterations) {
  # podział na train i test
  trainIndex <- createDataPartition(train$class, p = 0.9, list = FALSE)
  data_train <- train[trainIndex,]
  data_test <- train[-trainIndex,]
  
  # uczenie modelu + predict na zbiorze testowym
  for(j in 1:length(cps)) {
    treeModel <- rpart(class ~ ., data_train, cp = cps[j])
    treeProb <- predict(treeModel, newdata = data_test[,-ncol(data_test)], type = "prob")[,2]
    treePred <- predict(treeModel, newdata = data_test[,-ncol(data_test)], type = "class")
    
    precision_tree[i, j] <- Precision(y_pred = treePred, y_true = data_test$class)
    auc_tree[i, j] <- AUC(y_pred = treePred, y_true = data_test$class)
    precision_10_tree[i, j] <- prec_k(y_prob = treeProb, y_true = data_test$class, k = 0.1)
  }
}

# Wyznaczenie srednich wartosci miar dla poszczegolnych modeli
colMeans(precision_tree)     # 0.9571035 0.9581237 0.9591053 0.9591053 0.9591053 0.9591053 0.9591053 0.9591053 0.9591053 0.9591053
colMeans(auc_tree)           # 0.7148183 0.7217609 0.7283980 0.7283980 0.7283980 0.7283980 0.7283980 0.7283980 0.7283980 0.7283980
colMeans(precision_10_tree)  # 0.6168161 0.6150709 0.6139332 0.6139332 0.6139332 0.6139332 0.6139332 0.6139332 0.6139332 0.6139332

# Wyznaczenie modelu, dla ktorego wyszla najwieksza precyzja @ 10%
which.max(colMeans(precision_10_tree))  # model nr 1  => cp = 0.001
max(colMeans(precision_10_tree))        # 0.6168161


# Z przetestowanych modeli najlepiej dzialal model z parametrem cp = 0.001.


# Test nr 5 - wiecej iteracji z 'najlepszym' parametrem cp, tj cp = 0.001, w celu lepszego usrednienia wyniku jaki daje prec10%
number_of_iterations <- 100
precision_tree <- auc_tree <- precision_10_tree <- data.frame(matrix(0, nrow = number_of_iterations, ncol = 1))

for(i in 1:number_of_iterations) {
  # podział na train i test
  trainIndex <- createDataPartition(train$class, p = 0.9, list = FALSE)
  data_train <- train[trainIndex,]
  data_test <- train[-trainIndex,]
  
  # uczenie modelu + predict na zbiorze testowym
  treeModel <- rpart(class ~ ., data_train, cp = 0.001)
  treeProb <- predict(treeModel, newdata = data_test[,-ncol(data_test)], type = "prob")[,2]
  treePred <- predict(treeModel, newdata = data_test[,-ncol(data_test)], type = "class")
  
  precision_tree[i, 1] <- Precision(y_pred = treePred, y_true = data_test$class)
  auc_tree[i, 1] <- AUC(y_pred = treePred, y_true = data_test$class)
  precision_10_tree[i, 1] <- prec_k(y_prob = treeProb, y_true = data_test$class, k = 0.1)
}

colMeans(precision_tree)     # 0.9560567
colMeans(auc_tree)           # 0.7075907
colMeans(precision_10_tree)  # 0.598228


## adaboost ----------------------------------------------------------------------------------------------------------------

# Test nr 1
number_of_iterations <- 10
n_iter <- c(10, 30, 60, 100)
precision_ab <- auc_ab <- precision_10_ab <- data.frame(matrix(0, nrow = number_of_iterations, ncol = length(n_iter)))

for(i in 1:number_of_iterations) {
  # podział na train i test
  trainIndex <- createDataPartition(train$class, p = 0.9, list = FALSE)
  data_train <- train[trainIndex,]
  data_test <- train[-trainIndex,]
  
  # uczenie modelu + predict na zbiorze testowym
  for(j in 1:length(n_iter)) {
    adaboostModel <- adaboost(class ~ ., data_train, nIter = n_iter[j])
    adaboostProb <- predict(adaboostModel, newdata = data_test[,-ncol(data_test)])$prob[,2]
    adaboostPred <- predict(adaboostModel, newdata = data_test[,-ncol(data_test)])$class
    
    precision_ab[i, j] <- Precision(y_pred = adaboostPred, y_true = data_test$class)
    auc_ab[i, j] <- AUC(y_pred = adaboostPred, y_true = data_test$class)
    precision_10_ab[i, j] <- prec_k(y_prob = adaboostProb, y_true = data_test$class, k = 0.1)
  }
}

# Wyznaczenie srednich wartosci miar dla poszczegolnych modeli
colMeans(precision_ab)     # 0.9561109 0.9557523 0.9561452 0.9557099
colMeans(auc_ab)           # 0.7061943 0.7052700 0.7080591 0.7051088
colMeans(precision_10_ab)  # 0.3741828 0.3802500 0.3830000 0.3837500

# Wyznaczenie modelu, dla ktorego wyszla najwieksza precyzja @ 10%
which.max(colMeans(precision_10_ab))  # model nr 4  => nIter = 100
max(colMeans(precision_10_ab))        # 0.38375


# Test nr 2
number_of_iterations <- 10
n_iter <- c(150, 200, 270)
precision_ab2 <- auc_ab2 <- precision_10_ab2 <- data.frame(matrix(0, nrow = number_of_iterations, ncol = length(n_iter)))

for(i in 1:number_of_iterations) {
  # podział na train i test
  trainIndex <- createDataPartition(train$class, p = 0.9, list = FALSE)
  data_train <- train[trainIndex,]
  data_test <- train[-trainIndex,]
  
  # uczenie modelu + predict na zbiorze testowym
  for(j in 1:length(n_iter)) {
    adaboostModel <- adaboost(class ~ ., data_train, nIter = n_iter[j])
    adaboostProb <- predict(adaboostModel, newdata = data_test[,-ncol(data_test)])$prob[,2]
    adaboostPred <- predict(adaboostModel, newdata = data_test[,-ncol(data_test)])$class
    
    precision_ab2[i, j] <- Precision(y_pred = adaboostPred, y_true = data_test$class)
    auc_ab2[i, j] <- AUC(y_pred = adaboostPred, y_true = data_test$class)
    precision_10_ab2[i, j] <- prec_k(y_prob = adaboostProb, y_true = data_test$class, k = 0.1)
  }
}

# Wyznaczenie srednich wartosci miar dla poszczegolnych modeli
colMeans(precision_ab2)     # 0.9554208 0.9552994 0.9552585
colMeans(auc_ab2)           # 0.7033086 0.7024959 0.7022355
colMeans(precision_10_ab2)  # 0.39325 0.39175 0.38875

# Wyznaczenie modelu, dla ktorego wyszla najwieksza precyzja @ 10%
which.max(colMeans(precision_10_ab2))  # model nr 1  => nIter = 150
max(colMeans(precision_10_ab2))        # 0.39325


# Z przetestowanych modeli najlepiej dzialal model z parametrem nIter = 150.


## random forest -----------------------------------------------------------------------------------------------------------

# Test nr 1
number_of_iterations <- 10
n_trees <- seq(50, 500, 50)
precision_rf <- auc_rf <- precision_10_rf <- data.frame(matrix(0, nrow = number_of_iterations, ncol = length(n_trees)))

for(i in 1:number_of_iterations) {
  # podział na train i test
  trainIndex <- createDataPartition(train$class, p = 0.9, list = FALSE)
  data_train <- train[trainIndex,]
  data_test <- train[-trainIndex,]
  
  # uczenie modelu + predict na zbiorze testowym
  for(j in 1:length(n_trees)) {
    randomForestModel <- randomForest(class ~ ., data_train, ntree = n_trees[j])
    randomForestProb <- predict(randomForestModel, newdata = data_test[,-ncol(data_test)], type = 'prob')[,2]
    randomForestPred <- ifelse(randomForestProb < 0.5, 0, 1)
    
    precision_rf[i, j] <- Precision(y_pred = randomForestPred, y_true = data_test$class)
    auc_rf[i, j] <- AUC(y_pred = randomForestPred, y_true = data_test$class)
    precision_10_rf[i, j] <- prec_k(y_prob = randomForestProb, y_true = data_test$class, k = 0.1)
  }
}

# Wyznaczenie srednich wartosci miar dla kazdego lasu
colMeans(precision_rf)     # 0.9541034 0.9547565 0.9546594 0.9544083 0.9547832 0.9545080 0.9547356 0.9547371 0.9546881 0.9547084
colMeans(auc_rf)           # 0.6944491 0.6990209 0.6983653 0.6967310 0.6992185 0.6973144 0.6989042 0.6989177 0.6985764 0.6986931
colMeans(precision_10_rf)  # 0.4260891 0.4177474 0.4072919 0.4008654 0.3952965 0.4000553 0.3972786 0.3931825 0.3941966 0.3955647

# Wyznaczenie lasu, dla ktorego wyszla najwieksza precyzja @ 10%
which.max(colMeans(precision_10_rf))  # las nr 1 => 50 drzew
max(colMeans(precision_10_rf))        # 0.4260891

# Z przetestowanych lasow najlepszy wynik, wg precyzji 10%, osiagnal las o najmniejszej z testowanych liczb drzew, 
# tj. las o 50 drzewach - osiagnal on wartosc ww miary ~ 42,6%. Nastepny byl las o 100 drzewach, kolejny o 150, 
# a dla kolejnych wartosc precyzji @ 10% oscyluje wokol 40%. Sprawdze zatem jeszcze lasy o mniejszej liczbie drzew.


# Test nr 2
number_of_iterations <- 10
n_trees <- seq(10, 100, 20)
precision_rf2 <- auc_rf2 <- precision_10_rf2 <- data.frame(matrix(0, nrow = number_of_iterations, ncol = length(n_trees)))

for(i in 1:number_of_iterations) {
  # podział na train i test
  trainIndex <- createDataPartition(train$class, p = 0.9, list = FALSE)
  data_train <- train[trainIndex,]
  data_test <- train[-trainIndex,]
  
  # uczenie modelu + predict na zbiorze testowym
  for(j in 1:length(n_trees)) {
    randomForestModel <- randomForest(class ~ ., data_train, ntree = n_trees[j])
    randomForestProb <- predict(randomForestModel, newdata = data_test[,-ncol(data_test)], type = 'prob')[,2]
    randomForestPred <- ifelse(randomForestProb < 0.5, 0, 1)
    
    precision_rf2[i, j] <- Precision(y_pred = randomForestPred, y_true = data_test$class)
    auc_rf2[i, j] <- AUC(y_pred = randomForestPred, y_true = data_test$class)
    precision_10_rf2[i, j] <- prec_k(y_prob = randomForestProb, y_true = data_test$class, k = 0.1)
  }
}

# Wyznaczenie srednich wartosci miar dla kazdego lasu
colMeans(precision_rf2)     # 0.9533504 0.9539440 0.9549348 0.9546587 0.9544380
colMeans(auc_rf2)           # 0.6882664 0.6931777 0.6999631 0.6981987 0.6966763
colMeans(precision_10_rf2)  # 0.4083957 0.4591094 0.4132451 0.4154780 0.4125359

# Wyznaczenie lasu, dla ktorego wyszla najwieksza precyzja @ 10%
which.max(colMeans(precision_10_rf2))  # las nr 2 => 30 drzew
max(colMeans(precision_10_rf2))        # 0.4591094


# Test nr 3 - test lasow z takimi parametrami jak wyzej + wiecej iteracji
number_of_iterations <- 20
n_trees <- c(seq(10, 100, 20), seq(100, 500, 50))
precision_rf3 <- auc_rf3 <- precision_10_rf3 <- data.frame(matrix(0, nrow = number_of_iterations, ncol = length(n_trees)))

for(i in 1:number_of_iterations) {
  # podział na train i test
  trainIndex <- createDataPartition(train$class, p = 0.9, list = FALSE)
  data_train <- train[trainIndex,]
  data_test <- train[-trainIndex,]
  
  # uczenie modelu + predict na zbiorze testowym
  for(j in 1:length(n_trees)) {
    randomForestModel <- randomForest(class ~ ., data_train, ntree = n_trees[j])
    randomForestProb <- predict(randomForestModel, newdata = data_test[,-ncol(data_test)], type = 'prob')[,2]
    randomForestPred <- ifelse(randomForestProb < 0.5, 0, 1)
    
    precision_rf3[i, j] <- Precision(y_pred = randomForestPred, y_true = data_test$class)
    auc_rf3[i, j] <- AUC(y_pred = randomForestPred, y_true = data_test$class)
    precision_10_rf3[i, j] <- prec_k(y_prob = randomForestProb, y_true = data_test$class, k = 0.1)
  }
}

# Wyznaczenie srednich wartosci miar dla kazdego lasu
colMeans(precision_rf3)     
# 0.9538176 0.9537929 0.9540485 0.9544910 0.9542990 0.9544082 0.9548102 0.9544771 0.9547349 0.9544543 0.9547739 0.9548875 0.9549630 0.9546596 
colMeans(auc_rf3)           
# 0.6916777 0.6922260 0.6940336 0.6970918 0.6957873 0.6965981 0.6993396 0.6970291 0.6988074 0.6968854 0.6990836 0.6998785 0.7004107 0.6983451
colMeans(precision_10_rf3)  
# 0.4071290 0.4645380 0.4243067 0.4062949 0.4136400 0.4121720 0.4069800 0.4022042 0.4004592 0.4002692 0.3969988 0.3987880 0.3979293 0.3959613

# Wyznaczenie lasu, dla ktorego wyszla najwieksza precyzja @ 10%
which.max(colMeans(precision_10_rf3))  # las nr 2 => 30 drzew
max(colMeans(precision_10_rf3))        # 0.464538


# Przy ostatnich dwoch testach las z 30 drzewami okazal sie najlepszy wg miary prec10%, 
# sprawdze zatem wartosci w poblizu 30


# Test nr 4
number_of_iterations <- 20
n_trees <- seq(20, 40, 2)
precision_rf4 <- auc_rf4 <- precision_10_rf4 <- data.frame(matrix(0, nrow = number_of_iterations, ncol = length(n_trees)))

for(i in 1:number_of_iterations) {
  # podział na train i test
  trainIndex <- createDataPartition(train$class, p = 0.9, list = FALSE)
  data_train <- train[trainIndex,]
  data_test <- train[-trainIndex,]
  
  # uczenie modelu + predict na zbiorze testowym
  for(j in 1:length(n_trees)) {
    randomForestModel <- randomForest(class ~ ., data_train, ntree = n_trees[j])
    randomForestProb <- predict(randomForestModel, newdata = data_test[,-ncol(data_test)], type = 'prob')[,2]
    randomForestPred <- ifelse(randomForestProb < 0.5, 0, 1)
    
    precision_rf4[i, j] <- Precision(y_pred = randomForestPred, y_true = data_test$class)
    auc_rf4[i, j] <- AUC(y_pred = randomForestPred, y_true = data_test$class)
    precision_10_rf4[i, j] <- prec_k(y_prob = randomForestProb, y_true = data_test$class, k = 0.1)
  }
}

# Wyznaczenie srednich wartosci miar dla kazdego lasu
colMeans(precision_rf4)     
# 0.9531455 0.9522914 0.9526577 0.9528306 0.9525540 0.9526161 0.9526694 0.9528814 0.9527438 0.9529418 0.9531247 
colMeans(auc_rf4)           
# 0.6876494 0.6818275 0.6843039 0.6855977 0.6836666 0.6840662 0.6844480 0.6859659 0.6850432 0.6864017 0.6876613
colMeans(precision_10_rf4)  
# 0.4368806 0.4065558 0.4554204 0.4256021 0.4137616 0.4532046 0.4229640 0.4153076 0.4295557 0.4178161 0.4080054

# Wyznaczenie lasu, dla ktorego wyszla najwieksza precyzja @ 10%
which.max(colMeans(precision_10_rf4))  # las nr 3 => 24 drzewa
max(colMeans(precision_10_rf4))        # 0.4554204

plot(seq(20, 40, 2), colMeans(precision_10_rf4), type = "p")

# W tym tescie najlepszy wynik osiagnal las o 24 drzewach, niewiele gorszy wynik mial las o 30 drzewach. 


# Test nr 5  - wiecej iteracji z ntree = 30, w celu lepszego usrednienia wyniku jaki daje prec10%

number_of_iterations <- 100
precision_rf5 <- auc_rf5 <- precision_10_rf5 <- data.frame(matrix(0, nrow = number_of_iterations, ncol = 1))

for(i in 1:number_of_iterations) {
  # podział na train i test
  trainIndex <- createDataPartition(train$class, p = 0.9, list = FALSE)
  data_train <- train[trainIndex,]
  data_test <- train[-trainIndex,]
  
  # uczenie modelu + predict na zbiorze testowym
  randomForestModel <- randomForest(class ~ ., data_train, ntree = 30)
  randomForestProb <- predict(randomForestModel, newdata = data_test[,-ncol(data_test)], type = 'prob')[,2]
  randomForestPred <- ifelse(randomForestProb < 0.5, 0, 1)
  
  precision_rf5[i, 1] <- Precision(y_pred = randomForestPred, y_true = data_test$class)
  auc_rf5[i, 1] <- AUC(y_pred = randomForestPred, y_true = data_test$class)
  precision_10_rf5[i, 1] <- prec_k(y_prob = randomForestProb, y_true = data_test$class, k = 0.1)
}

colMeans(precision_rf5)     # 0.9541741
colMeans(auc_rf5)           # 0.6948647
colMeans(precision_10_rf5)  # 0.4619902


# Test nr 6 - czy lepszy jest las o 24, czy o 30 drzewach?

number_of_iterations <- 100
n_trees <- c(24, 30)
precision_rf6 <- auc_rf6 <- precision_10_rf6 <- data.frame(matrix(0, nrow = number_of_iterations, ncol = length(n_trees)))

for(i in 1:number_of_iterations) {
  # podział na train i test
  trainIndex <- createDataPartition(train$class, p = 0.9, list = FALSE)
  data_train <- train[trainIndex,]
  data_test <- train[-trainIndex,]
  
  # uczenie modelu + predict na zbiorze testowym
  for(j in 1:length(n_trees)) {
    randomForestModel <- randomForest(class ~ ., data_train, ntree = n_trees[j])
    randomForestProb <- predict(randomForestModel, newdata = data_test[,-ncol(data_test)], type = 'prob')[,2]
    randomForestPred <- ifelse(randomForestProb < 0.5, 0, 1)
    
    precision_rf6[i, j] <- Precision(y_pred = randomForestPred, y_true = data_test$class)
    auc_rf6[i, j] <- AUC(y_pred = randomForestPred, y_true = data_test$class)
    precision_10_rf6[i, j] <- prec_k(y_prob = randomForestProb, y_true = data_test$class, k = 0.1)
  }
}

# Wyznaczenie srednich wartosci miar dla kazdego lasu
colMeans(precision_rf6)     # 0.9537269 0.9538165
colMeans(auc_rf6)           # 0.6917355 0.6923673
colMeans(precision_10_rf6)  # 0.4716005 0.4575528 

# Wyznaczenie lasu, dla ktorego wyszla najwieksza precyzja @ 10%
which.max(colMeans(precision_10_rf6))  # las nr 1 => 24 drzewa
max(colMeans(precision_10_rf6))        # 0.4716005


## xgboost -----------------------------------------------------------------------------------------------------------------

# Test nr 1
number_of_iterations <- 10
n_rounds <- seq(1, 10, 1)
precision_xgb <- auc_xgb <- precision_10_xgb <- data.frame(matrix(0, nrow = number_of_iterations, ncol = length(n_rounds)))

for(i in 1:number_of_iterations) {
  # podział na train i test
  train_numeric_Index <- createDataPartition(train_numeric$class, p = 0.9, list = FALSE)
  data_train_numeric <- train_numeric[train_numeric_Index,]
  data_test_numeric <- train_numeric[-train_numeric_Index,]
  
  # uczenie modelu + predict na zbiorze testowym
  for(j in 1:length(n_rounds)) {
    dtrain <- xgb.DMatrix(data = data.matrix(data_train_numeric[,-ncol(data_train_numeric)]), label= data_train_numeric$class)
    dtest <- xgb.DMatrix(data = data.matrix(data_test_numeric[,-ncol(data_test_numeric)]), label= data_test_numeric$class)
    
    xgboostModel <- xgboost(data = dtrain, nrounds = n_rounds[j], num_class = 2, objective = "multi:softprob")
    xgboostProb <- predict(xgboostModel, dtest)
    xgboostProb <- matrix(xgboostProb, ncol = 2, byrow = T)[,2]
    xgboostPred <- ifelse(xgboostProb < 0.5, 0, 1)
    
    precision_xgb[i, j] <- Precision(y_pred = xgboostPred, y_true = data_test_numeric$class)
    auc_xgb[i, j] <- AUC(y_pred = xgboostPred, y_true = data_test_numeric$class)
    precision_10_xgb[i, j] <- prec_k(y_prob = xgboostProb, y_true = data_test_numeric$class, k = 0.1)
  }
}

# Wyznaczenie srednich wartosci miar dla kazdego modelu
colMeans(precision_xgb)     # 0.9585601 0.9582442 0.9583385 0.9581865 0.9580947 0.9581995 0.9581560 0.9581392 0.9581951 0.9582943
colMeans(auc_xgb)           # 0.7227836 0.7208704 0.7214809 0.7204158 0.7198389 0.7205682 0.7203610 0.7202187 0.7206402 0.7213173
colMeans(precision_10_xgb)  # 0.5763273 0.5351728 0.4794761 0.4565755 0.4545713 0.4071287 0.4078519 0.4109477 0.4114158 0.4149578

# Wyznaczenie modelu, dla ktorego wyszla najwieksza precyzja @ 10%
which.max(colMeans(precision_10_xgb))  # model nr 1 => nrounds = 1
max(colMeans(precision_10_xgb))        # 0.5763273

# Zgodnie z oczekiwaniami po wczesniejszych malych testach, najlepsze wyniki (wg miary prec10%) wyszly dla najmniejszych wartosci
# parametru nround. Powtorze zatem testy dla nround z przedzialu [1, 5], tylko z wieksza liczba iteracji w celu lepszego 
# usrednienia wynikow prec10%.


# Test nr 2
number_of_iterations <- 20
n_rounds <- seq(1, 5, 1)
precision_xgb <- auc_xgb <- precision_10_xgb <- data.frame(matrix(0, nrow = number_of_iterations, ncol = length(n_rounds)))

for(i in 1:number_of_iterations) {
  # podział na train i test
  train_numeric_Index <- createDataPartition(train_numeric$class, p = 0.9, list = FALSE)
  data_train_numeric <- train_numeric[train_numeric_Index,]
  data_test_numeric <- train_numeric[-train_numeric_Index,]
  
  # uczenie modelu + predict na zbiorze testowym
  for(j in 1:length(n_rounds)) {
    dtrain <- xgb.DMatrix(data = data.matrix(data_train_numeric[,-ncol(data_train_numeric)]), label= data_train_numeric$class)
    dtest <- xgb.DMatrix(data = data.matrix(data_test_numeric[,-ncol(data_test_numeric)]), label= data_test_numeric$class)
    
    xgboostModel <- xgboost(data = dtrain, nrounds = n_rounds[j], num_class = 2, objective = "multi:softprob")
    xgboostProb <- predict(xgboostModel, dtest)
    xgboostProb <- matrix(xgboostProb, ncol = 2, byrow = T)[,2]
    xgboostPred <- ifelse(xgboostProb < 0.5, 0, 1)
    
    precision_xgb[i, j] <- Precision(y_pred = xgboostPred, y_true = data_test_numeric$class)
    auc_xgb[i, j] <- AUC(y_pred = xgboostPred, y_true = data_test_numeric$class)
    precision_10_xgb[i, j] <- prec_k(y_prob = xgboostProb, y_true = data_test_numeric$class, k = 0.1)
  }
}

# Wyznaczenie srednich wartosci miar dla kazdego modelu
colMeans(precision_xgb)     # 0.9571261 0.9570302 0.9571380 0.9571185 0.9570369
colMeans(auc_xgb)           # 0.7144908 0.7138747 0.7147360 0.7146119 0.7141083
colMeans(precision_10_xgb)  # 0.6879942 0.5043046 0.4452818 0.4224497 0.4245068

# Wyznaczenie modelu, dla ktorego wyszla najwieksza precyzja @ 10%
which.max(colMeans(precision_10_xgb))  # model nr 1 => nrounds = 1
max(colMeans(precision_10_xgb))        # 0.6879942


# Test nr 3 - to samo co w tescie nr 2, dodatkowo eta = 0.1
number_of_iterations <- 20
n_rounds <- seq(1, 5, 1)
precision_xgb <- auc_xgb <- precision_10_xgb <- data.frame(matrix(0, nrow = number_of_iterations, ncol = length(n_rounds)))

for(i in 1:number_of_iterations) {
  # podział na train i test
  train_numeric_Index <- createDataPartition(train_numeric$class, p = 0.9, list = FALSE)
  data_train_numeric <- train_numeric[train_numeric_Index,]
  data_test_numeric <- train_numeric[-train_numeric_Index,]
  
  # uczenie modelu + predict na zbiorze testowym
  for(j in 1:length(n_rounds)) {
    dtrain <- xgb.DMatrix(data = data.matrix(data_train_numeric[,-ncol(data_train_numeric)]), label= data_train_numeric$class)
    dtest <- xgb.DMatrix(data = data.matrix(data_test_numeric[,-ncol(data_test_numeric)]), label= data_test_numeric$class)
    
    xgboostModel <- xgboost(data = dtrain, nrounds = n_rounds[j], eta = 0.1, num_class = 2, objective = "multi:softprob")
    xgboostProb <- predict(xgboostModel, dtest)
    xgboostProb <- matrix(xgboostProb, ncol = 2, byrow = T)[,2]
    xgboostPred <- ifelse(xgboostProb < 0.5, 0, 1)
    
    precision_xgb[i, j] <- Precision(y_pred = xgboostPred, y_true = data_test_numeric$class)
    auc_xgb[i, j] <- AUC(y_pred = xgboostPred, y_true = data_test_numeric$class)
    precision_10_xgb[i, j] <- prec_k(y_prob = xgboostProb, y_true = data_test_numeric$class, k = 0.1)
  }
}

# Wyznaczenie srednich wartosci miar dla kazdego modelu
colMeans(precision_xgb)     # 0.9584324 0.9583613 0.9582513 0.9584019 0.9583566
colMeans(auc_xgb)           # 0.7201728 0.7197676 0.7190319 0.7200236 0.7197189
colMeans(precision_10_xgb)  # 0.6292644 0.5615619 0.4914466 0.4630945 0.4538593

# Wyznaczenie modelu, dla ktorego wyszla najwieksza precyzja @ 10%
which.max(colMeans(precision_10_xgb))  # model nr 1 => nrounds = 1
max(colMeans(precision_10_xgb))        # 0.6292644


# Test nr 4 - to samo co w tescie nr 2, dodatkowo eta = 0.01
number_of_iterations <- 20
n_rounds <- seq(1, 5, 1)
precision_xgb <- auc_xgb <- precision_10_xgb <- data.frame(matrix(0, nrow = number_of_iterations, ncol = length(n_rounds)))

for(i in 1:number_of_iterations) {
  # podział na train i test
  train_numeric_Index <- createDataPartition(train_numeric$class, p = 0.9, list = FALSE)
  data_train_numeric <- train_numeric[train_numeric_Index,]
  data_test_numeric <- train_numeric[-train_numeric_Index,]
  
  # uczenie modelu + predict na zbiorze testowym
  for(j in 1:length(n_rounds)) {
    dtrain <- xgb.DMatrix(data = data.matrix(data_train_numeric[,-ncol(data_train_numeric)]), label= data_train_numeric$class)
    dtest <- xgb.DMatrix(data = data.matrix(data_test_numeric[,-ncol(data_test_numeric)]), label= data_test_numeric$class)
    
    xgboostModel <- xgboost(data = dtrain, nrounds = n_rounds[j], eta = 0.01, num_class = 2, objective = "multi:softprob")
    xgboostProb <- predict(xgboostModel, dtest)
    xgboostProb <- matrix(xgboostProb, ncol = 2, byrow = T)[,2]
    xgboostPred <- ifelse(xgboostProb < 0.5, 0, 1)
    
    precision_xgb[i, j] <- Precision(y_pred = xgboostPred, y_true = data_test_numeric$class)
    auc_xgb[i, j] <- AUC(y_pred = xgboostPred, y_true = data_test_numeric$class)
    precision_10_xgb[i, j] <- prec_k(y_prob = xgboostProb, y_true = data_test_numeric$class, k = 0.1)
  }
}

# Wyznaczenie srednich wartosci miar dla kazdego modelu
colMeans(precision_xgb)     # 0.9564632 0.9565652 0.9565627 0.9566013 0.9565405
colMeans(auc_xgb)           # 0.7153264 0.7160198 0.7159824 0.7162410 0.7158576
colMeans(precision_10_xgb)  # 0.6384626 0.6339222 0.6309437 0.6313607 0.6202509

# Wyznaczenie modelu, dla ktorego wyszla najwieksza precyzja @ 10%
which.max(colMeans(precision_10_xgb))  # model nr 1 => nrounds = 1
max(colMeans(precision_10_xgb))        # 0.6384626


# Test nr 5 - testowanie eta, nrounds = 1
number_of_iterations <- 10
etas <- c(0.01, 0.05, 0.1, 0.2, 0.5)
precision_xgb <- auc_xgb <- precision_10_xgb <- data.frame(matrix(0, nrow = number_of_iterations, ncol = length(etas)))

for(i in 1:number_of_iterations) {
  # podział na train i test
  train_numeric_Index <- createDataPartition(train_numeric$class, p = 0.9, list = FALSE)
  data_train_numeric <- train_numeric[train_numeric_Index,]
  data_test_numeric <- train_numeric[-train_numeric_Index,]
  
  # uczenie modelu + predict na zbiorze testowym
  for(j in 1:length(etas)) {
    dtrain <- xgb.DMatrix(data = data.matrix(data_train_numeric[,-ncol(data_train_numeric)]), label= data_train_numeric$class)
    dtest <- xgb.DMatrix(data = data.matrix(data_test_numeric[,-ncol(data_test_numeric)]), label= data_test_numeric$class)
    
    xgboostModel <- xgboost(data = dtrain, nrounds = 1, eta = etas[j], num_class = 2, objective = "multi:softprob")
    xgboostProb <- predict(xgboostModel, dtest)
    xgboostProb <- matrix(xgboostProb, ncol = 2, byrow = T)[,2]
    xgboostPred <- ifelse(xgboostProb < 0.5, 0, 1)
    
    precision_xgb[i, j] <- Precision(y_pred = xgboostPred, y_true = data_test_numeric$class)
    auc_xgb[i, j] <- AUC(y_pred = xgboostPred, y_true = data_test_numeric$class)
    precision_10_xgb[i, j] <- prec_k(y_prob = xgboostProb, y_true = data_test_numeric$class, k = 0.1)
  }
}

# Wyznaczenie srednich wartosci miar dla kazdego modelu
colMeans(precision_xgb)     # 0.957956 0.957956 0.957956 0.957956 0.957956
colMeans(auc_xgb)           # 0.7195429 0.7195429 0.7195429 0.7195429 0.7195429 
colMeans(precision_10_xgb)  # 0.6151127 0.6151127 0.6151127 0.6151127 0.6151127

# Wniosek: parametr eta nie ma wplywu na model


# Z przetestowanych modeli najlepiej dzialal model z parametrem nrounds = 1.


# Test nr 5 - wiecej iteracji z 'najlepszym' parametrem, tj. nrounds = 1, w celu lepszego usrednienia wyniku jaki daje prec10%
number_of_iterations <- 100
precision_xgb <- auc_xgb <- precision_10_xgb <- data.frame(matrix(0, nrow = number_of_iterations, ncol = 1))

for(i in 1:number_of_iterations) {
  # podział na train i test
  train_numeric_Index <- createDataPartition(train_numeric$class, p = 0.9, list = FALSE)
  data_train_numeric <- train_numeric[train_numeric_Index,]
  data_test_numeric <- train_numeric[-train_numeric_Index,]
  
  # uczenie modelu + predict na zbiorze testowym
  dtrain <- xgb.DMatrix(data = data.matrix(data_train_numeric[,-ncol(data_train_numeric)]), label= data_train_numeric$class)
  dtest <- xgb.DMatrix(data = data.matrix(data_test_numeric[,-ncol(data_test_numeric)]), label= data_test_numeric$class)
  
  xgboostModel <- xgboost(data = dtrain, nrounds = 1, num_class = 2, objective = "multi:softprob")
  xgboostProb <- predict(xgboostModel, dtest)
  xgboostProb <- matrix(xgboostProb, ncol = 2, byrow = T)[,2]
  xgboostPred <- ifelse(xgboostProb < 0.5, 0, 1)
  
  precision_xgb[i, 1] <- Precision(y_pred = xgboostPred, y_true = data_test_numeric$class)
  auc_xgb[i, 1] <- AUC(y_pred = xgboostPred, y_true = data_test_numeric$class)
  precision_10_xgb[i, 1] <- prec_k(y_prob = xgboostProb, y_true = data_test_numeric$class, k = 0.1)
}

colMeans(precision_xgb)     # 0.9571346
colMeans(auc_xgb)           # 0.7160975
colMeans(precision_10_xgb)  # 0.6197034


#---------------------------------------------------------------------------------------------------------------------------

# Ze wszystkich przeprowadzonych testow najlepszym modelem okazal sie xgboost z parametrem nrounds = 1

#---------------------------------------------------------------------------------------------------------------------------

### Predykcja dla danych ze zbioru testx

## Ujednolicenie zbiorow train i testx

# Sprawdzenie, na jakim etapie sa oczyszczone dane ze zbioru testx

nrow(train)  # 40000
ncol(train)  # 112
head(train)

nrow(testx)  # 10000
ncol(testx)  # 111
head(testx)

colnames(train)[-ncol(train)] == colnames(testx)  # w train i w testx pozostaly te same zmienne


# Zamiana factorow na wartosci liczbowe (poniewaz xgboost wymaga wartosci liczbowych)

class(train$class)  # "factor" ; musze zatem jeszcze raz zaladowac plik train.txt i jeszcze raz puscic oczyszczanie danych, 
                    # zeby nie popsuc danych zawartych w tej zmiennej

# (dalsze operacje po ponownym zaladowaniu pliku train.txt i oczyszczeniu danych)

# W celu odpowiedniej zamiany factorow na wartosci liczbowe zlacze najpierw oba te zbiory w jeden
train_testx <- rbind.data.frame(train[,-ncol(train)], testx)
head(train_testx)
nrow(train_testx)  # 50000
ncol(train_testx)  # 111

# Wyznaczenie zmiennych typu factor
train_testx_types_of_columns <- sapply(train_testx, function(x) {class(x)})
train_testx_factor_col_names <- colnames(train_testx[train_testx_types_of_columns == "factor"])

# Zmiana factorow na wartosci liczbowe
train_testx[colnames(train_testx) %in% train_testx_factor_col_names] <- sapply(train_testx[colnames(train_testx) %in% train_testx_factor_col_names], function(x) {as.numeric(x)})


## Przygotowanie danych do xgboost
dtrain <- xgb.DMatrix(data = data.matrix(train_testx[1:nrow(train),]), label= train$class)
dtest <- xgb.DMatrix(data = data.matrix(train_testx[(nrow(train) + 1) : nrow(train_testx),]))


## Uczenie modelu
xgboostModel <- xgboost(data = dtrain, nrounds = 1, num_class = 2, objective = "multi:softprob")


## Predykcja na zbiorze testowym
xgboostProb <- predict(xgboostModel, dtest)
xgboostProb <- matrix(xgboostProb, ncol = 2, byrow = T)[,2]


## Zapis do pliku wynikow predykcji
results <- c("\"AGAPAL\"", xgboostProb)
write(results, file = "AGAPAL.txt")
