# Stop and Wait Protocol Simulation using NS2

# Create simulator object
set ns [new Simulator]
$ns color 1 Blue

# Open the NAM output file
set nf [open out.nam w]
$ns namtrace-all $nf

# Finish procedure
proc finish {} {
    global ns nf
    $ns flush-trace
    close $nf
    exec nam out.nam &
    exit 0
}

# Create sender and receiver nodes
set n0 [$ns node]
set n1 [$ns node]
$ns at 0.0 "$n0 label \"Sender\""
$ns at 0.0 "$n1 label \"Receiver\""

# Duplex link
$ns duplex-link $n0 $n1 1Mb 200ms DropTail
$ns duplex-link-op $n0 $n1 orient right

# TCP agent setup
set tcp [new Agent/TCP]
$tcp set fid_ 1
$tcp set window_ 1
$tcp set maxcwnd_ 1
$ns attach-agent $n0 $tcp
set sink [new Agent/TCPSink]
$ns attach-agent $n1 $sink
$ns connect $tcp $sink

# FTP over TCP
set ftp [new Application/FTP]
$ftp attach-agent $tcp

# Schedule events
$ns at 0.5 "$ftp start"
$ns at 4.0 "finish"

# Run simulation
$ns run
