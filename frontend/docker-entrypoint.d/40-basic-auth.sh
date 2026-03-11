#!/bin/sh
set -eu

HTPASSWD_FILE=/etc/nginx/.htpasswd
USERS=${BASIC_AUTH_USERS:-}
PASSWORD=${BASIC_AUTH_PASSWORD:-}

if [ -z "$USERS" ] || [ -z "$PASSWORD" ]; then
  echo "BASIC_AUTH_USERS and BASIC_AUTH_PASSWORD must be set" >&2
  exit 1
fi

rm -f "$HTPASSWD_FILE"
first_user=1
old_ifs=$IFS
IFS=,
for raw_user in $USERS; do
  IFS=$old_ifs
  user=$(printf '%s' "$raw_user" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
  IFS=,
  if [ -z "$user" ]; then
    continue
  fi

  if [ "$first_user" -eq 1 ]; then
    htpasswd -bc "$HTPASSWD_FILE" "$user" "$PASSWORD"
    first_user=0
  else
    htpasswd -b "$HTPASSWD_FILE" "$user" "$PASSWORD"
  fi
done
IFS=$old_ifs

if [ "$first_user" -eq 1 ]; then
  echo "No valid BASIC_AUTH_USERS entries were provided" >&2
  exit 1
fi