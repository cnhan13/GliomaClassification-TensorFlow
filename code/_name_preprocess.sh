find . -type f -name "*OT*.mha" -execdir mv -v -- {} "ot.mha" \;
find . -type f -name "*Flair*.mha" -execdir mv -v -- {} "flair.mha" \;
find . -type f -name "*T2*.mha" -execdir mv -v -- {} "t2.mha" \;
find . -type f -name "*T1c*.mha" -execdir mv -v -- {} "t1c.mha" \;
find . -type f -name "*T1*.mha" -execdir mv -v -- {} "t1.mha" \;
find . -type f -name "*.mha" -execdir mv -v -- {} .. \;

find . -type d -name "VSD*" -exec rm -rf {} +

find HGG/ -type d -name "brats*" -printf "mv -v %h/%f %h/../H_%f\n" | bash
find LGG/ -type d -name "brats*" -printf "mv -v %h/%f %h/../L_%f\n" | bash
rm -rf HGG LGG
