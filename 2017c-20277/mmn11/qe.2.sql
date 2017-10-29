select cid, rno, sum(price * amount) as total_price
from receipt
natural join "café"
natural join buys
natural join sells
where
    cname = 'abc' and
    date_part('month', rdate) = '07' and date_part('year', rdate) = '2017'
group by cid, rno