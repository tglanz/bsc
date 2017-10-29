select distinct on (r1.cid)
	name
from
	receipt as r1
    	natural join item
        natural join client,
    receipt as r2
where
	itype = 'hot drink' and
	r1.cid = r2.cid and
    r1.rdate = r2.rdate and
    r1.rno <> r2.rno and
    r1.license <> r2.license