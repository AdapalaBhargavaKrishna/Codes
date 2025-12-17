# Star Topology
set ns [new Simulator]

# Trace File
set nf [open star.nam w]
$ns namtrace-all $nf

proc finish {} {
    global ns nf
    $ns flush-trace
    close $nf
    exec nam star.nam &
    exit 0
}

# Nodes
set center [$ns node]
set n1 [$ns node]
set n2 [$ns node]
set n3 [$ns node]
set n4 [$ns node]

# Links
$ns duplex-link $center $n1 1Mb 10ms DropTail
$ns duplex-link $center $n2 1Mb 10ms DropTail
$ns duplex-link $center $n3 1Mb 10ms DropTail
$ns duplex-link $center $n4 1Mb 10ms DropTail

# Orientation
$ns duplex-link-op $center $n1 orient left
$ns duplex-link-op $center $n2 orient right
$ns duplex-link-op $center $n3 orient up
$ns duplex-link-op $center $n4 orient down

# TCP & Sink between center and n1
set tcp [new Agent/TCP]
$ns attach-agent $center $tcp
set sink [new Agent/TCPSink]
$ns attach-agent $n1 $sink
$ns connect $tcp $sink

# FTP
set ftp [new Application/FTP]
$ftp attach-agent $tcp
$ns at 0.5 "$ftp start"
$ns at 4.0 "$ftp stop"
$ns at 5.0 "finish"
$ns run