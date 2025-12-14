Step-by-step guide to set up PostgreSQL on Ubuntu

1. Install PostgreSQL 
```
sudo apt update
sudo apt install postgresql postgresql-contrib -y
```

Check status:
```
sudo systemctl status postgresql
```
You should see active (running).

Start/enable if needed:
```
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

2. Switch to the postgres user
PostgreSQL creates a system user postgres by default:
```
sudo -i -u postgres
```

Then access PostgreSQL shell:
```psql```

You’ll see postgres=# prompt.

3. Create database and user
Inside psql:
```
-- create user
CREATE USER myuser WITH PASSWORD 'mypassword';

-- create database
CREATE DATABASE mydatabase;

-- give privileges
GRANT ALL PRIVILEGES ON DATABASE mydatabase TO myuser;
```

Optional: allow myuser to create tables, etc.: ALTER USER myuser CREATEDB;

Exit psql: 
```
\q
exit
```

4. Configure PostgreSQL to allow password authentication
Open PostgreSQL config for client authentication:
```
sudo nano /etc/postgresql/16/main/pg_hba.conf
```

Version may vary (16 → your version)
Find line:
local   all             all                                     peer

Change peer → md5:
local   all             all                                     md5

Save and exit (Ctrl+O, Ctrl+X).

Then restart PostgreSQL:
```
sudo systemctl restart postgresql
```

5. privilege on schema public
Log in as postgres
```
sudo -i -u postgres
psql
```
Switch to your database
```
\c your_table
```
Grant the missing privileges (THIS is the key)
```
-- allow using the schema
GRANT USAGE ON SCHEMA public TO semyon;

-- allow creating tables
GRANT CREATE ON SCHEMA public TO semyon;
```
6. Test connection
```
psql -h localhost -U myuser -d mydatabase
```