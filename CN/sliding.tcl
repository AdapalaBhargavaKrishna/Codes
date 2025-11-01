# Sliding Window Protocol Simulation using NS2
set ns [new Simulator]

# Output files
set namf [open sliding_window.nam w]
$ns namtrace-all $namf

set trf [open sliding_window.tr w]
$ns trace-all $trf

# Finish procedure
proc finish {} {
    global ns namf trf
    $ns flush-trace
    close $namf
    close $trf
    exec nam sliding_window.nam &
    exit 0
}

# Create nodes
set n0 [$ns node]
set n1 [$ns node]

$ns duplex-link $n0 $n1 0.2Mb 200ms DropTail
$ns duplex-link-op $n0 $n1 orient right

# TCP agent setup
set tcp [new Agent/TCP]
set sink [new Agent/TCPSink]
$ns attach-agent $n0 $tcp
$ns attach-agent $n1 $sink
$ns connect $tcp $sink

# FTP over TCP
set ftp [new Application/FTP]
$ftp attach-agent $tcp

# Configure sliding window
$tcp set windowInit_ 4
$tcp set maxcwnd_ 4

# Schedule events
$ns at 0.1 "$ftp start"
$ns at 5.0 "finish"

# Run the simulation
$ns run