for x in *; do

    if [ $x = 'src' ] ; then
        echo "$x is src. continue"
        continue


    elif [[ $x == *.sh ]] ; then
        echo "$x is bash. continue"
        continue

    # elif [ -d $x ] ; then
    #     echo "$x is a directory"

    # elif [[ $x == *.html ]] ; then
    #     echo "$x is html"
    fi

    rm -rf $x
done
