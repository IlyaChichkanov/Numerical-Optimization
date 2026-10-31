#!/bin/bash
# SessionStart-хук облачных сессий Claude Code: подтягивает закрытые инструкции
# курса из приватного репозитория и выводит их в контекст сессии.
#
# Самих инструкций в этом публичном репозитории нет. Хук читает токен из
# переменной окружения COURSE_PRIVATE_TOKEN (задаётся в настройках облачного
# окружения), скачивает файлы в /opt/course-private и кладёт CLAUDE.md в корень
# клона как CLAUDE.local.md, добавив его в .git/info/exclude. Закрытые skill-ы
# (.claude/skills/<имя>/ приватного репозитория) так же копируются в
# .claude/skills/ клона и исключаются из git. Вне облака хук ничего не делает.
# Падать не должен: любой исход завершается кодом 0.
#
# Проверить ветку приватного репозитория до слияния в main:
# COURSE_PRIVATE_BRANCH=<ветка> при запуске хука.

[ "${CLAUDE_CODE_REMOTE:-}" = "true" ] || exit 0

PRIVATE_REPO="IlyaChichkanov/numopt-private"
BRANCH="${COURSE_PRIVATE_BRANCH:-main}"
FILES="CLAUDE.md docs/prompts.md docs/tooling.md"
# файлы skill-ов по одному: новый файл skill — добавить сюда
SKILL_FILES=".claude/skills/lecture/SKILL.md .claude/skills/lecture/reference/why.md
  .claude/skills/lecture/scripts/sentence_lengths.py .claude/skills/lecture/scripts/katex_check.js
  .claude/skills/lecture/scripts/check_notebooks.py"
DEST="/opt/course-private"
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-/home/user/Numerical-Optimization}"

fetch() {  # fetch <путь в репозитории> <куда сохранить>; 0 при успехе
  local f="$1" out="$2" code
  code=$(curl -sS --max-time 20 -o "$out" -w '%{http_code}' \
           -H "Authorization: token $COURSE_PRIVATE_TOKEN" \
           "https://raw.githubusercontent.com/$PRIVATE_REPO/$BRANCH/$f") || code=000
  [ "$code" = "200" ] && return 0
  # запасной путь: REST API отдаёт содержимое файла напрямую
  code=$(curl -sS --max-time 20 -o "$out" -w '%{http_code}' \
           -H "Authorization: Bearer $COURSE_PRIVATE_TOKEN" \
           -H "Accept: application/vnd.github.raw+json" \
           "https://api.github.com/repos/$PRIVATE_REPO/contents/$f?ref=$BRANCH") || code=000
  [ "$code" = "200" ] && return 0
  echo "private-instructions: $f не получен (HTTP $code)" >&2
  return 1
}

if [ -n "${COURSE_PRIVATE_TOKEN:-}" ]; then
  mkdir -p "$DEST"
  for f in $FILES $SKILL_FILES; do
    mkdir -p "$DEST/$(dirname "$f")"
    if fetch "$f" "$DEST/$f.tmp"; then mv "$DEST/$f.tmp" "$DEST/$f"; else rm -f "$DEST/$f.tmp"; fi
  done
else
  echo "private-instructions: переменная COURSE_PRIVATE_TOKEN не задана" >&2
fi

exclude() {  # exclude <строка>: добавить в .git/info/exclude клона, если её там нет
  grep -qxF "$1" "$PROJECT_DIR/.git/info/exclude" 2>/dev/null || echo "$1" >> "$PROJECT_DIR/.git/info/exclude"
}

SKILLS=""
if [ -d "$DEST/.claude/skills" ] && [ -d "$PROJECT_DIR/.git" ]; then
  for s in "$DEST"/.claude/skills/*/; do
    [ -d "$s" ] || continue
    name=$(basename "$s")
    mkdir -p "$PROJECT_DIR/.claude/skills/$name" && cp -R "$s." "$PROJECT_DIR/.claude/skills/$name/"
    exclude "/.claude/skills/$name/"
    SKILLS="$SKILLS $name"
  done
fi

if [ -f "$DEST/CLAUDE.md" ]; then
  if [ -d "$PROJECT_DIR/.git" ]; then
    cp "$DEST/CLAUDE.md" "$PROJECT_DIR/CLAUDE.local.md"
    exclude 'CLAUDE.local.md'
  fi
  # stdout SessionStart-хука попадает в контекст сессии
  echo "Закрытые инструкции курса (источник: $PRIVATE_REPO; полные тексты в $DEST):"
  echo
  cat "$DEST/CLAUDE.md"
  [ -n "$SKILLS" ] && { echo; echo "Закрытые skill-ы курса скопированы в .claude/skills/:$SKILLS"; }
else
  echo "Закрытые инструкции курса недоступны: проверьте COURSE_PRIVATE_TOKEN в настройках облачного окружения и права токена (Contents: read) на $PRIVATE_REPO."
fi
exit 0
