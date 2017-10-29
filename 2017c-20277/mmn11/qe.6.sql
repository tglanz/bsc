with all_items as (
    select iname
    from item
)
select distinct on (cid)
	name
from buys as b
    natural join client
where not exists (
    select iname
    from all_items
    where iname not in (
        select iname
        from buys
        where cid = b.cid
    )
)