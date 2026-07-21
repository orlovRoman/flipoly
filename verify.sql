SELECT asset, count(*), sum(case when is_active then 1 else 0 end) as active from model_registry group by asset order by asset;
