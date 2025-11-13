#!/bin/bash
set -euo pipefail

# 1) Source the venv if it exists
if [ -f /app/.venv/bin/activate ]; then
    . /app/.venv/bin/activate
fi

# 2) Ensure MySQL directories exist and have correct ownership
mkdir -p /var/lib/mysql /var/run/mysqld
chown -R mysql:mysql /var/lib/mysql /var/run/mysqld

# 3) Initialize MySQL if system tables missing (dev only)
if [ ! -d /var/lib/mysql/mysql ]; then
    echo "[entrypoint] initializing MySQL datadir..."
    mysqld --initialize-insecure --user=mysql --datadir=/var/lib/mysql
fi

# 4) Start MySQL in background
echo "[entrypoint] starting mysqld_safe..."
nohup mysqld_safe --datadir=/var/lib/mysql &>/tmp/mysqld.log &

# Give it a moment to start
sleep 5

# 5) Create dev user if it doesn't exist
mysql -u root <<-EOSQL
    CREATE USER IF NOT EXISTS 'dev'@'%' IDENTIFIED BY 'devpass';
    GRANT ALL PRIVILEGES ON *.* TO 'dev'@'%';
    FLUSH PRIVILEGES;
EOSQL

# Optional: check MySQL status
mysqladmin ping -u root &>/dev/null || echo "[entrypoint] Warning: MySQL did not start correctly"

# 6) Drop into interactive shell
exec /bin/bash

