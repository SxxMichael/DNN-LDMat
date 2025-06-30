#setwd("C:/Users/xiaox/Desktop/Academic/Research/CAR-Continuous/Simulations/data_1")
ped <- read.table("ped.ped", sep = "\t")
info <- read.table("info.info", header = T)

rare_id <- which(info$maf < 0.001)  # 4436
common_snp <- ped[, -rare_id]
common_snp_info <- info[-rare_id,]

# standardization
p <- ncol(common_snp)
common_snp_scaled <- 1/sqrt(p) * apply(common_snp, 2, scale)

write.table(common_snp_scaled, "SNP.txt", col.names = common_snp_info$SNP)

# create labels
n <- nrow(ped)
func_frq <- 0.3 # 30% are causal snps
causal_snp_id <- sample(1:p, floor(p*func_frq))
noise <- rnorm(n)
snp_effect <- common_snp_scaled[,causal_snp_id] %*% 
  rnorm(length(causal_snp_id), 0, 0.6)
y <- snp_effect + noise
write.table(y, "response.txt")


###################################
# Summary LD matrices
##################################
common_snp_scaled <- read.table("SNP.txt", header = T)
common_snp_scaled <- as.matrix(common_snp_scaled)
y <- read.table("response.txt")
y <- as.matrix(y)

n <- length(y)
n_train <- floor(n*0.8)
n_test <- n-n_train
train_id <- sample(1:n,n_train)
test_id <- setdiff(1:n, train_id)

LD_mat_train <- crossprod(common_snp_scaled[train_id,])/n_train
LD_mat_test <- crossprod(common_snp_scaled[-train_id,])/n_test
write.table(LD_mat_train, "LDMatrix_train.txt", col.names = common_snp_info$SNP)
write.table(LD_mat_test, "LDMatrix_test.txt", col.names = common_snp_info$SNP)

y_train <- y[train_id]
y_test <- y[test_id]
write.table(y_train, "response_train.txt")
write.table(y_test, "response_test.txt")

