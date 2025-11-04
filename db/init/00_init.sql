-- 00_init.sql
CREATE SCHEMA IF NOT EXISTS users;
CREATE TABLE IF NOT EXISTS users.customer (
    id SERIAL PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL
);
