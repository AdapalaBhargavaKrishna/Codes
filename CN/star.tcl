# Star Topology Simulation using NS2

# Create Simulator instance
set ns [new Simulator]

# Color flows for NAM visualization
$ns color 1 blue
$ns color 2 red

# Enable Distance Vector routing protocol
$ns rtproto DV

# Open NAM trace file
set nf [open out.nam w]
$ns namtrace-all $nf

# Define finish procedure
proc finish {} {
    global ns nf
    $ns flush-trace
    close $nf
    exec nam out.nam &
    exit 0
}

# Create nodes: n(0) is the central hub
for {set i 0} {$i < 7} {incr i} {
    set n($i) [$ns node]
}

# Connect hub to all peripherals
for {set i 1} {$i < 7} {incr i} {
    $ns duplex-link $n(0) $n($i) 512Kb 10ms SFQ
}

# Orientation for visualization
$ns duplex-link-op $n(0) $n(1) orient left-up
$ns duplex-link-op $n(0) $n(2) orient right-up
$ns duplex-link-op $n(0) $n(3) orient right
$ns duplex-link-op $n(0) $n(4) orient right-down
$ns duplex-link-op $n(0) $n(5) orient left-down
$ns duplex-link-op $n(0) $n(6) orient left

# TCP Setup (n1 —(TCP)—> n4)
set tcp0 [new Agent/TCP]
$tcp0 set class_ 1
$ns attach-agent $n(1) $tcp0
set sink0 [new Agent/TCPSink]
$ns attach-agent $n(4) $sink0
$ns connect $tcp0 $sink0
set ftp0 [new Application/FTP]
$ftp0 attach-agent $tcp0

# UDP & CBR Setup (n2 —(UDP/CBR)—> n5)
set udp0 [new Agent/UDP]
$udp0 set class_ 2
$ns attach-agent $n(2) $udp0
set null0 [new Agent/Null]
$ns attach-agent $n(5) $null0
$ns connect $udp0 $null0
set cbr0 [new Application/Traffic/CBR]
$cbr0 set rate_ 256Kb
$cbr0 attach-agent $udp0

# Link Failure/Recovery Events
$ns rtmodel-at 0.5 down $n(0) $n(5)
$ns rtmodel-at 0.9 up $n(0) $n(5)
$ns rtmodel-at 0.7 down $n(0) $n(4)
$ns rtmodel-at 1.2 up $n(0) $n(4)

# Schedule Application Events
$ns at 0.1 "$ftp0 start"
$ns at 1.5 "$ftp0 stop"
$ns at 0.2 "$cbr0 start"
$ns at 1.3 "$cbr0 stop"
$ns at 2.0 "finish"

# Run the simulation
$ns run
