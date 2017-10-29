select name
from client
where client.cid in (
    select cid
    from receipt
    group by cid
    having count (rno) > 500
)

/* 
    -- or maybe it's this ????
    -- the question is not really clear

select name
from client
where client.cid in (
    select distinct(cid)
    from buys
    where not exists (
        select *
        from likes
        where buys.cid = likes.cid
    )
    group by cid, iname
    having sum (amount) > 500
)

*/