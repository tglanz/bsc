create or replace function get_count_of_item_buys(
    _cid numeric(5, 0),
    _iname varchar(30)
) returns integer as $$
    declare
    begin
        return count(*)
            from buys
            where cid = _cid and iname = _iname;
    end;
$$ language plpgsql;

create or replace function update_likes_of_client(
    _cid numeric(5, 0),
    _iname varchar(30),
    _is_like boolean
) returns void as $$
    declare
        is_currently_like boolean;
    begin
        select count(*) > 0 into is_currently_like
        from likes
        where cid = _cid and iname = _iname;

        if is_currently_like and not _is_like then
            delete
            from likes
            where cid = _cid and iname = _iname;
        elsif not is_currently_like and _is_like then
            insert into likes
            values (_cid, _iname); 
        end if;
    end;
$$ language plpgsql;

create or replace function trigf() returns trigger as $$
    declare
        is_like boolean;
        threshold integer;
    begin

        threshold = 2;

        -- if the update or insert a new item
        select get_count_of_item_buys(NEW.cid, NEW.iname) >= threshold into is_like;
        perform update_likes_of_client(NEW.cid, NEW.iname, is_like);

        if TG_OP = 'UPDATE' then
            -- if because of the update, the old item is removed
            select get_count_of_item_buys(OLD.cid, OLD.iname) >= threshold into is_like;
            perform update_likes_of_client(OLD.cid, OLD.iname, is_like);
        end if; 

        return NEW;
    end;
$$ language plpgsql;

drop trigger if exists T on buys;

create trigger T after update or insert on buys for each row execute procedure trigf();
