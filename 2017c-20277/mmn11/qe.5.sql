with total_amounts_of_clients_that_bought as (
    select *
    from (
        select distinct cid
        from receipt as r1
        where 'cappuccino' in (
            select iname
            from receipt as r2
            natural join buys
            where r1.cid = r2.cid
        )
    ) as relevant_clients
    natural join (
        select cid, sum (amount) as total_amount
        from receipt
        natural join buys
        group by cid
    ) as total_amounts
)
select name
from total_amounts_of_clients_that_bought
natural join client
where total_amount = (
    select max (total_amount)
    from total_amounts_of_clients_that_bought
)