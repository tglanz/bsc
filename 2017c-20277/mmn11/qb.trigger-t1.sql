create or replace function trigf1() returns trigger as $$
    declare
    begin
        if count(*) > 0
            from receipt natural join sells
            where cid = NEW.cid and rno = NEW.rno and iname = NEW.iname
        then
            return NEW;
        end if;

        raise notice 'No item named `%` is for sell :(', NEW.iname;
        return null;
    end;
$$ language plpgsql;

drop trigger if exists T1 on buys;
create trigger T1 before insert on buys for each row execute procedure trigf1();