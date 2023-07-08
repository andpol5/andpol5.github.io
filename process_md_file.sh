#!/bin/bash

MY_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SITE_PATH=$MY_PATH/site

fullpath=$1
dirpath=$( dirname $1 )
sourcefile=$( basename $1 )
targetfile=$(echo "$sourcefile" | cut -f 1 -d '.')'.html'

# Linux: Using file system information to get creation and modicfication date.
# Warning: This may yield faulty data if you use git and multiple computers.
#creationDate=$( stat -c %w $fullpath | cut -f 1 -d ' ' )
#lastUpdated=$( stat -c %y $fullpath | cut -f 1 -d ' ' )

# Use git to get creation and modification date
lastUpdated=$( git log -1 --format="%ci" -- $fullpath | cut -f 1 -d ' ' )
creationDate=$( git log --format="%ai" -- $fullpath | tail -1 | cut -f 1 -d ' ' )

echo Processing: $fullpath

rm $dirpath/$targetfile
pandoc $fullpath -o $dirpath/$targetfile --standalone \
	--css "/css/milligram.min.css" --css "/css/custom.css" \
	--template=templates/easy_template.html \
	--variable=lastUpdated:$lastUpdated --variable=creationDate:$creationDate

exit 0
