# Sliding Window Protocol
set ns [new Simulator]

# Trace Files
set nf [open out.nam w]
$ns namtrace-all $nf

proc finish {} {
    global ns nf
    $ns flush-trace
    close $nf
    exec nam out.nam &
    exit 0
}

# Nodes
set n0 [$ns node]
set n1 [$ns node]
$ns duplex-link $n0 $n1 1Mb 200ms DropTail
$ns duplex-link-op $n0 $n1 orient right

# TCP & Sink
set tcp [new Agent/TCP]
$tcp set fid_ 1
$tcp set window_ 4
$tcp set maxcwnd_ 4
$ns attach-agent $n0 $tcp

set sink [new Agent/TCPSink]
$ns attach-agent $n1 $sink
$ns connect $tcp $sink

# FTP
set ftp [new Application/FTP]
$ftp attach-agent $tcp

# Events
$ns at 0.5 "$ftp start"
$ns at 3.0 "$ftp stop"
$ns at 4.0 "finish"

$ns run