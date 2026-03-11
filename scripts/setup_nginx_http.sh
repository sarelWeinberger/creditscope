#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
NGINX_CONF_SOURCE="$PROJECT_ROOT/deploy/nginx/creditscope.conf"
NGINX_CONF_TARGET="/etc/nginx/sites-available/creditscope"
NGINX_ENABLED_TARGET="/etc/nginx/sites-enabled/creditscope"
SSL_CERT_TARGET="/etc/ssl/certs/creditscope-selfsigned.crt"
SSL_KEY_TARGET="/etc/ssl/private/creditscope-selfsigned.key"

if ! command -v sudo >/dev/null 2>&1; then
    echo "sudo is required" >&2
    exit 1
fi

export DEBIAN_FRONTEND=noninteractive

PUBLIC_IP=${PUBLIC_IP:-$(curl -4 -s https://ifconfig.me 2>/dev/null || hostname -I | awk '{print $1}')}

if [ -z "$PUBLIC_IP" ]; then
    echo "Unable to determine PUBLIC_IP" >&2
    exit 1
fi

if ! command -v nginx >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y nginx
fi

sudo mkdir -p /etc/nginx/sites-available /etc/nginx/sites-enabled
sudo cp "$NGINX_CONF_SOURCE" "$NGINX_CONF_TARGET"

if [ ! -f "$SSL_CERT_TARGET" ] || [ ! -f "$SSL_KEY_TARGET" ]; then
    tmp_openssl_config=$(mktemp)
    cat > "$tmp_openssl_config" <<EOF
[req]
distinguished_name = req_distinguished_name
x509_extensions = v3_req
prompt = no

[req_distinguished_name]
CN = $PUBLIC_IP

[v3_req]
subjectAltName = @alt_names

[alt_names]
IP.1 = $PUBLIC_IP
EOF

    sudo openssl req \
        -x509 \
        -nodes \
        -newkey rsa:2048 \
        -days 365 \
        -keyout "$SSL_KEY_TARGET" \
        -out "$SSL_CERT_TARGET" \
        -config "$tmp_openssl_config"
    rm -f "$tmp_openssl_config"
    sudo chmod 600 "$SSL_KEY_TARGET"
fi

if [ -L "$NGINX_ENABLED_TARGET" ] || [ -e "$NGINX_ENABLED_TARGET" ]; then
    sudo rm -f "$NGINX_ENABLED_TARGET"
fi
sudo ln -s "$NGINX_CONF_TARGET" "$NGINX_ENABLED_TARGET"

if [ -e /etc/nginx/sites-enabled/default ]; then
    sudo rm -f /etc/nginx/sites-enabled/default
fi

sudo nginx -t

if command -v systemctl >/dev/null 2>&1; then
    sudo systemctl enable nginx
    sudo systemctl reload nginx 2>/dev/null || sudo systemctl restart nginx
elif command -v service >/dev/null 2>&1; then
    sudo service nginx reload 2>/dev/null || sudo service nginx restart
else
    sudo nginx -s reload 2>/dev/null || sudo nginx
fi

echo "nginx is serving CreditScope on ports 80 and 443 for $PUBLIC_IP"