create table item (
	iname varchar(30) primary key,
    itype varchar(30)
);

create table "café" (
	license numeric(5, 0) primary key,
    cname varchar(30),
    address varchar(30)
);

create table client (
	cid numeric(5, 0) primary key,
    name varchar(30),
    phone numeric(9, 0)
);

create table likes (
	cid numeric(5, 0) references client(cid),
    iname varchar(30) references item(iname),
    primary key (cid, iname)
);

create table sells (
	license numeric(5, 0) references "café"(license),
    iname varchar(30) references item(iname),
    price float not null check (price > 0),
    primary key (license, iname)
);

create table receipt (
	cid numeric(5, 0) references client(cid),
    rno numeric(5, 0) not null,
    license numeric(5, 0) references "café"(license),
    rdate date,
    primary key (cid, rno)
);

create table buys (
	cid numeric(5, 0),
    rno numeric(5, 0),
    iname varchar(30) references item(iname),
    amount int not null check (amount > 0),
    primary key (cid, rno, iname),
    foreign key (cid, rno) references receipt
);