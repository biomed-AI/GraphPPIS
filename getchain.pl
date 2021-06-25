#!/usr/bin/perl

if (@ARGV != 2){die "usage: $0 PDB_file_path PDBID(e.g. 3mcbA)\n";}
else {$pdbdir_local = $ARGV[0]; getchain($ARGV[1]);}
sub getchain{
	my ($pdbnm) = @_;
	if(length($pdbnm) != 5) {
		die "irregular pdbnm (5 chars required): $pdbnm\n";
	}
	$bn = substr($pdbnm, 0, 4);
	$pn0 = "$pdbdir_local/pdb$bn.ent.gz";
	if(-e $pn0){
		open(fp, "gunzip -c $pn0|");
	} elsif(-e "$pdbdir_local/$bn.pdb.gz"){
		open(fp, "gunzip -c $pdbdir_local/$bn.pdb.gz|");
	} elsif(-e "$pdbdir_local/$bn.pdb"){
		open(fp, "$pdbdir_local/$bn.pdb");
	} else{
		print "no file: $pn0\n"; return;
	}
	open(fpo, "> $pdbnm");
	$chn = $chn0 = substr($pdbnm, 4, 1);
#	$chn =~ tr/[a-z]/[A-Z]/;
	$na = 0;
	while($line = <fp>){
		last if $line =~ /^ENDMDL/;
		if($line =~ /^HETATM/){
			$line = &replace_het($line);
		}
		next if not $line =~ /^ATOM /;
#
		my $rn = substr($line, 17, 3);
		if($rn eq 'ASX') {substr($line, 17, 3) = 'ASN'}
		if($rn eq 'GLX') {substr($line, 17, 3) = 'GLN'}
# remove H atom
		my $an = substr($line, 12, 2);
		next if $an =~ /^H|^ H/;
#
		next if substr($line, 16, 1) !~ /[ A1]/;
		substr($line, 16, 1) = ' ';
# remove duplicate atoms
		$ct = substr($line, 21, 1);
		$chn = $ct if $chn eq "_";
		die "multiple chains in $pn0: $chn $ct\n" if($chn0 eq "_" and $chn ne $chn);
		next if $chn ne $ct;
		chomp $line;
		print fpo substr($line, 0, 80), "\n";
		$na ++;
	}
	print fpo "TER\n";
#	print "$pdbnm: $na\n";
	close fp; close fpo;
}
my $info_old = "";
sub replace_het{
	my ($str) = @_; my $replace = 0;
	my $str0 = $str;
	if(substr($str, 17, 3) eq "MSE") {
		$replace = 1;
		substr($str, 17, 3) = "MET";
		substr($str, 12, 3) = " SD" if(substr($str, 12, 3) eq "SE ");
	}elsif(substr($str, 17, 3) =~ /CS.|CCS/) {
		$replace = 1;
		substr($str, 17, 3) = "CYS";
	}
	if($replace){
		if($info_old ne substr($str0, 17, 10)){
			$info_old = substr($str0, 17, 10);
#			print "replacing $info_old\n";
		}
		substr($str, 0, 6) = "ATOM  ";
	}
	return $str;
}
