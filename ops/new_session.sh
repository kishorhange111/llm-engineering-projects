# Create a session that is NOT in asia-southeast1 (every runtime lost today was in that region).
# Usage: bash ops/new_session.sh <name> <A100|L4|T4> [max_tries]
C=~/.local/bin/colab
N=$1; G=$2; TRIES=${3:-4}
for i in $(seq 1 $TRIES); do
  $C new -s $N --gpu $G >/dev/null 2>&1 || { echo "allocation refused ($i)"; sleep 20; continue; }
  ep=$($C sessions 2>/dev/null | grep "^\[$N\]" | awk '{print $2}')
  case "$ep" in
    *-ass1*) echo "got $ep (asia-southeast1) - releasing and retrying"; $C stop -s $N >/dev/null 2>&1; sleep 10 ;;
    "") echo "session not listed ($i)"; sleep 10 ;;
    *) echo "OK $N -> $ep"; exit 0 ;;
  esac
done
echo "FAILED to get a non-asia-southeast1 $G"; exit 1
