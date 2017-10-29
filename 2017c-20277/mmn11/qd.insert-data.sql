insert into item values
('cappuccino', 'hot drink'),
('macchiato', 'hot drink'),
('orange juice', 'cold drink');

insert into "café" values
(12345, 'abc', '1 palm street'),
(22346, 'cba', '2 canal road'),
(65432, 'aaa', '3 hill drive');

insert into client values
(1, 'hadas', 111111),
(2, 'danny', 222222),
(3, 'yossi', 444444);

insert into sells values
(12345, 'cappuccino', 5.5),
(12345, 'orange juice', 7),
(22346, 'macchiato', 8),
(22346, 'orange juice', 4.5),
(65432, 'cappuccino', 10);

insert into receipt values
(1, 1, 12345, '7/1/2017'),
(2, 1, 22346, '7/8/2017'),
(3, 1, 65432, '7/20/2017'),
(1, 2, 12345, '7/22/2017');

insert into buys values
(1, 1, 'cappuccino', 2),
(2, 1, 'macchiato', 2),
(2, 1, 'orange juice', 4),
(1, 2, 'cappuccino', 2),
(1, 2, 'orange juice', 1);