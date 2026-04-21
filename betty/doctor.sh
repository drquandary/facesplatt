#!/usr/bin/env bash
# Betty harness pre-flight. Validates every known roadblock from PLAYBOOK.md.
# Exit 0 = ready to run. Exit 1 = fix the first FAIL before proceeding.
set -u

BETTY_HOST="${BETTY_HOST:-login.betty.parcc.upenn.edu}"
BETTY_USER="${BETTY_USER:-jvadala}"
BETTY_ROOT_DEFAULT="/vast/home/j/${BETTY_USER}/facesplatt"
BETTY_ROOT="${BETTY_ROOT:-$BETTY_ROOT_DEFAULT}"

pass() { printf "  \033[32m✓\033[0m %s\n" "$1"; }
fail() { printf "  \033[31m✗\033[0m %s\n" "$1"; FAILED=1; }
warn() { printf "  \033[33m!\033[0m %s\n" "$1"; }
hdr()  { printf "\n\033[1m%s\033[0m\n" "$1"; }

FAILED=0

hdr "1. Kerberos ticket"
if klist -s 2>/dev/null; then
  principal=$(klist 2>/dev/null | awk '/Principal:/ {print $2}')
  expires=$(klist 2>/dev/null | awk '/krbtgt/ {print $3, $4, $5, $6}')
  pass "ticket for $principal, expires $expires"
else
  fail "no ticket — run: kinit ${BETTY_USER}@UPENN.EDU"
fi

hdr "2. Penn network reachability"
if ping -c 1 -W 2 "$BETTY_HOST" >/dev/null 2>&1; then
  pass "$BETTY_HOST reachable (on Penn network or VPN)"
else
  warn "$BETTY_HOST not reachable via ping — may just be ICMP blocked. If SSH also fails, connect to Penn VPN."
fi

hdr "3. SSH host key known"
if grep -q "$BETTY_HOST" ~/.ssh/known_hosts 2>/dev/null; then
  pass "$BETTY_HOST in ~/.ssh/known_hosts"
else
  fail "host key missing — run: ssh-keyscan -T 10 $BETTY_HOST >> ~/.ssh/known_hosts"
fi

hdr "4. SSH config for GSSAPI + ControlMaster"
if grep -q "Host $BETTY_HOST" ~/.ssh/config 2>/dev/null; then
  if grep -A10 "Host $BETTY_HOST" ~/.ssh/config | grep -q "ControlMaster"; then
    pass "~/.ssh/config has ControlMaster stanza"
  else
    fail "~/.ssh/config has host but no ControlMaster — see PLAYBOOK R3"
  fi
else
  fail "no ~/.ssh/config entry for $BETTY_HOST — see PLAYBOOK R2/R3"
fi

hdr "5. SSH ControlMaster live"
if ssh -O check "$BETTY_HOST" 2>&1 | grep -q "Master running"; then
  pass "control socket live (no re-auth needed)"
else
  fail "no live master — run: ssh $BETTY_HOST 'echo ok'  (answer Duo once)"
fi

if [ "$FAILED" = 1 ]; then
  printf "\n\033[31mFAIL — fix the above before continuing.\033[0m\n"
  exit 1
fi

hdr "6. Remote checks"
remote_report=$(ssh "$BETTY_HOST" bash -s <<EOF 2>&1
set -u
BETTY_ROOT="$BETTY_ROOT"

# Lmod sourceable?
if command -v module >/dev/null 2>&1; then
  echo "LMOD_LOGIN=ok"
else
  if [ -f /etc/profile.d/Z98-lmod.sh ]; then
    source /etc/profile.d/Z98-lmod.sh 2>/dev/null
    command -v module >/dev/null && echo "LMOD_SOURCE=ok" || echo "LMOD=missing"
  else
    echo "LMOD=missing"
  fi
fi

# Required modules
if command -v module >/dev/null 2>&1; then
  for m in anaconda3/2023.09-0 cuda/12.8.1 gcc/13.3.0; do
    module spider "\$m" >/dev/null 2>&1 && echo "MODULE_OK=\$m" || echo "MODULE_MISSING=\$m"
  done
fi

# Quota
if [ -x /vast/parcc/sw/bin/parcc_quota.py ]; then
  line=\$(/vast/parcc/sw/bin/parcc_quota.py 2>/dev/null | awk -F'|' '/vast/ {gsub(/^ +| +\$/,"",\$4); gsub(/^ +| +\$/,"",\$5); print \$4" / "\$5}' | head -1)
  echo "QUOTA=\$line"
else
  echo "QUOTA=missing"
fi

# BETTY_ROOT writable
mkdir -p "\$BETTY_ROOT" 2>/dev/null
if [ -w "\$BETTY_ROOT" ]; then echo "ROOT_OK=\$BETTY_ROOT"; else echo "ROOT_FAIL=\$BETTY_ROOT"; fi

# sinfo
command -v sinfo >/dev/null 2>&1 && echo "SLURM=ok" || echo "SLURM=missing"
EOF
)

if grep -q "LMOD_LOGIN=ok\|LMOD_SOURCE=ok" <<<"$remote_report"; then
  pass "Lmod available (module command resolves)"
else
  fail "Lmod not resolvable — scripts must source /etc/profile.d/Z98-lmod.sh"
fi

for m in anaconda3/2023.09-0 cuda/12.8.1 gcc/13.3.0; do
  if grep -q "MODULE_OK=$m" <<<"$remote_report"; then
    pass "module available: $m"
  else
    fail "module missing: $m"
  fi
done

q=$(grep "^QUOTA=" <<<"$remote_report" | head -1 | cut -d= -f2-)
if [ -n "$q" ] && [ "$q" != "missing" ]; then
  pass "home quota: $q"
else
  warn "quota check failed — parcc_quota.py unavailable"
fi

if grep -q "ROOT_OK=" <<<"$remote_report"; then
  pass "BETTY_ROOT writable: $BETTY_ROOT"
else
  fail "BETTY_ROOT not writable: $BETTY_ROOT"
fi

if grep -q "SLURM=ok" <<<"$remote_report"; then
  pass "slurm commands available"
else
  fail "sinfo missing — wrong node?"
fi

printf "\n"
if [ "$FAILED" = 1 ]; then
  printf "\033[31mNOT READY — fix failures above.\033[0m See PLAYBOOK.md for each roadblock.\n"
  exit 1
else
  printf "\033[32mREADY.\033[0m\n"
  printf "Next: bash sync_up.sh ; then on Betty: bash setup_facelift.sh ; sbatch run_facelift.sbatch\n"
  exit 0
fi
