// Package main is the entrypoint for the Podstack scheduler plugin.
// Phase 2: Currently a stub that registers the GPU-aware scheduler plugin.
package main

import (
	"fmt"
	"os"

	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"

	"github.com/podstack/serverless/internal/scheduler"
)

func main() {
	ctrl.SetLogger(zap.New(zap.UseDevMode(true)))
	log := ctrl.Log.WithName("scheduler")

	// Phase 2: This will register as a K8s scheduler framework plugin.
	// For now, it just initializes the plugin and exits.
	plugin := scheduler.NewPlugin("http://prometheus.podstack-system:9090", log)

	log.Info("Podstack GPU scheduler plugin initialized (Phase 2 stub)",
		"pluginName", plugin.Name(),
	)

	fmt.Fprintf(os.Stderr, "Podstack GPU scheduler plugin %s is a Phase 2 feature.\n", plugin.Name())
	fmt.Fprintf(os.Stderr, "It will be integrated with the K8s scheduler framework in Phase 2.\n")
}
