drop table if exists ml.atl_sizemodel_cust_sizes;
create table ml.atl_sizemodel_cust_sizes
(
	customer_id int,
	model_timestamp timestamp,
	size_object jsonb
);


drop table if exists ml.atl_sizemodel_article_sizes;
create table ml.atl_sizemodel_article_sizes
(
	article_id varchar(100),
	model_timestamp timestamp,
	size_object jsonb
);