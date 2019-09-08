# Load ced.csv and edit_dist.csv matrices from /dist_matrices repertory

ced <- as.dist(ced)
edit_dist <- as.dist(edit_dist)
cedhc <- hclust(ced, method="ward.D2")
edhc <- hclust(edit_dist, method="ward.D2")
A2Rplot(cedhc, k = 4, boxes = TRUE, col.down = c("#0000FF", "#000000", "#7F00FF", "#FF7F00"))
A2Rplot(edhc, k = 5, boxes = TRUE, col.down = c("#0000FF", "#000000", "#BBBBBB", "#FF0000", "#71CB68"))